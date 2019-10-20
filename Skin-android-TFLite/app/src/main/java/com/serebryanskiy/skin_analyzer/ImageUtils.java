package com.serebryanskiy.skin_analyzer;

import android.graphics.Bitmap;

public class ImageUtils {

    public static Bitmap prepareImageForClassification(Bitmap bitmap) {
        Bitmap bmpGrayscale = Bitmap.createScaledBitmap(
                bitmap,
                ModelConfig.INPUT_IMG_SIZE_WIDTH,
                ModelConfig.INPUT_IMG_SIZE_HEIGHT,
                false);
        return bmpGrayscale;
    }

}
