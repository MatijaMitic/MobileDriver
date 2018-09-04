package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class TensorFlowLaneMatcher {

    private static final int WANTED_WIDTH = 160;
    private static final int WANTED_HEIGHT = 80;
    private static final String MODEL_FILE = "file:///android_asset/keras_K_frozen.pb";
    private static final String INPUT_NODE = "batch_normalization_1_input_1";
    private static final String OUTPUT_NODE = "Final_1/Relu";
    private static final String IMAGE_NAME = "pug1.jpeg";

    private TensorFlowInferenceInterface mInferenceInterface;

    private AssetManager assetManager;

    public TensorFlowLaneMatcher(AssetManager assets){
        assetManager = assets;
        mInferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
    }

    public Bitmap detectLane(Bitmap inputBitmap){

        int[] intValues = new int[WANTED_WIDTH * WANTED_HEIGHT];
        float[] floatValues = new float[WANTED_WIDTH * WANTED_HEIGHT * 3];
        float[] outputValues = new float[WANTED_WIDTH * WANTED_HEIGHT];

        Bitmap scaledBitmap = Bitmap.createScaledBitmap(inputBitmap, WANTED_WIDTH, WANTED_HEIGHT, true);
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = ((val >> 16) & 0x00FF);
            floatValues[i * 3 + 1] = ((val >> 8) & 0x00FF);
            floatValues[i * 3 + 2] = (val & 0x00FF);
        }

        mInferenceInterface.feed(INPUT_NODE, floatValues, 1, WANTED_HEIGHT, WANTED_WIDTH, 3);
        mInferenceInterface.run(new String[] {OUTPUT_NODE}, false);
        mInferenceInterface.fetch(OUTPUT_NODE, outputValues);

        for (int i = 0; i < intValues.length; ++i) {
            intValues[i] = 0xFF000000
                    | (((int) (outputValues[i ] * 0)) << 16)
                    | (((int) (outputValues[i ] * 255)) << 8)
                    | ((int)  (outputValues[i ] * 0));
        }

        Bitmap outputBitmap = scaledBitmap.copy( scaledBitmap.getConfig() , true);
        outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());

        putOverlay(scaledBitmap,outputBitmap);

        return Bitmap.createScaledBitmap(scaledBitmap, inputBitmap.getWidth(), inputBitmap.getHeight(), true);
    }

    public void putOverlay(Bitmap bitmap, Bitmap overlay) {
        Canvas canvas = new Canvas(bitmap);
        Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
        paint.setAlpha(75);
        canvas.drawBitmap(overlay, 0, 0, paint);
    }

}
