package com.example.sttapp;


import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Map;


public class MainActivity extends AppCompatActivity implements Runnable {
    private static final String TAG = MainActivity.class.getName();

    private Module mModuleEncoder;
    private TextView mTextView;
    private Button mButton;

    private Sonopy spectrogram;

    private final static String[] tokens = {"'",
            " ",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "_",  };
    private final static int AUDIO_LEN_LIMIT = 2;

    private final static int REQUEST_RECORD_AUDIO = 13;
    private final static int SAMPLE_RATE = 8000;
    private final static int RECORDING_LENGTH = SAMPLE_RATE * AUDIO_LEN_LIMIT;

    private int mStart = 1;
    private HandlerThread mTimerThread;
    private Handler mTimerHandler;
    private final Runnable mRunnable = new Runnable() {
        @Override
        public void run() {
            mTimerHandler.postDelayed(mRunnable, 1000);

            MainActivity.this.runOnUiThread(
                    () -> {
                        mButton.setText(String.format("Listening - %ds left", AUDIO_LEN_LIMIT - mStart));
                        mStart += 1;
                    });
        }
    };

    @Override
    protected void onDestroy() {
        stopTimerThread();
        super.onDestroy();
    }

    protected void stopTimerThread() {
        mTimerThread.quitSafely();
        try {
            mTimerThread.join();
            mTimerThread = null;
            mTimerHandler = null;
            mStart = 1;
        } catch (InterruptedException e) {
            Log.e(TAG, "Error on stopping background thread", e);
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mButton = findViewById(R.id.btnRecognize);
        mTextView = findViewById(R.id.tvResult);
        spectrogram = new Sonopy();

        mButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButton.setText(String.format("Listening - %ds left", AUDIO_LEN_LIMIT));
                mButton.setEnabled(false);

                Thread thread = new Thread(MainActivity.this);
                thread.start();

                mTimerThread = new HandlerThread("Timer");
                mTimerThread.start();
                mTimerHandler = new Handler(mTimerThread.getLooper());
                mTimerHandler.postDelayed(mRunnable, 1000);

            }
        });
        requestMicrophonePermission();
    }

    private void requestMicrophonePermission() {
        requestPermissions(
                new String[]{android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
    }

    private String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e(TAG, assetName + ": " + e.getLocalizedMessage());
        }
        return null;
    }

    private void showTranslationResult(String result) {
        mTextView.setText(result);
    }

    public void run() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.DEFAULT, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                bufferSize);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            return;
        }
        record.startRecording();

        long shortsRead = 0;
        int recordingOffset = 0;
        short[] audioBuffer = new short[bufferSize / 2];
        short[] recordingBuffer = new short[RECORDING_LENGTH];

        while (shortsRead < RECORDING_LENGTH) {
            int numberOfShort = record.read(audioBuffer, 0, audioBuffer.length);
            shortsRead += numberOfShort;
            System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, numberOfShort);
            recordingOffset += numberOfShort;
        }

        record.stop();
        record.release();
        stopTimerThread();

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mButton.setText("Recognizing...");
            }
        });

        double[] doubleInputBuffer = new double[RECORDING_LENGTH];

        for (int i = 0; i < RECORDING_LENGTH; ++i) {
            doubleInputBuffer[i] = recordingBuffer[i] / (double) Short.MAX_VALUE;
        }

        float[] melspectrogram = spectrogram.process(doubleInputBuffer);
        final String result = recognize(melspectrogram);

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                showTranslationResult(result);
                mButton.setEnabled(true);
                mButton.setText("Start");
            }
        });
    }

    private String recognize(float[] floatInputBuffer) {
        if (mModuleEncoder == null) {
            final String moduleFileAbsoluteFilePath = new File(
                    assetFilePath(this, "speechrecognition.pt")).getAbsolutePath();
            mModuleEncoder = Module.load(moduleFileAbsoluteFilePath);
        }
        FloatBuffer zeros = Tensor.allocateFloatBuffer(81*1024);
        for(int i = 0;i < 81*1024;i++){
            zeros.put(0f);
        }
        FloatBuffer inTensorBuffer = Tensor.allocateFloatBuffer(81*81*10);
        for(int i = 0 ; i < 81*81*10;i++) {
            if(i<floatInputBuffer.length) {
                inTensorBuffer.put(floatInputBuffer[i]);
            }else{
                inTensorBuffer.put(0f);
            }
        }

        Tensor hidden = Tensor.fromBlob(zeros,new long[]{1, 81, 1024});

        Tensor inTensor = Tensor.fromBlob(inTensorBuffer, new long[]{81,81,10});


        IValue[] returned = mModuleEncoder.forward(IValue.from(inTensor), IValue.tupleFrom(IValue.from(hidden),IValue.from(hidden))).toTuple();
        Tensor tensor = returned[0].toTensor();
        final float[] values = tensor.getDataAsFloatArray();
        final float[][] rows = unsqueeze(values);
        StringBuilder result = new StringBuilder();
        for (float[] row : rows) {
            double[] softmaxvalues = softmax(row);
            int index = argmax(softmaxvalues);
            Log.i("INDEX", index + "");
            result.append(tokens[index]);
        }
       return result.toString();
    }

    private float[][] unsqueeze(float[] values){
        int numRows = values.length/tokens.length;
        float[][] array = new float[numRows][tokens.length];
        for(int i=0;i<numRows;i++){
            System.arraycopy(values, tokens.length * i, array[i], 0, tokens.length);
        }

        return array;
    }


    private double[] softmax(float[] array){
        double[] output = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            output[i] = Double.parseDouble(String.valueOf(array[i]));

        }
        double total = Arrays.stream(output).map(x -> Math.exp(Math.abs(x))).sum();
        
        return Arrays.stream(output).map(x -> Math.abs(x)/total).toArray();
    }

    private int argmax(double[] array) {
        int maxIdx = 0;
        double maxVal = -Double.MAX_VALUE;
        for (int j = 0; j < array.length; j++) {
            if (array[j] > maxVal){
                maxVal = array[j];
                maxIdx = j;
            }
        }
        return maxIdx;
    }
}
