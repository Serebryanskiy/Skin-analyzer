package com.serebryanskiy.skin_analyzer;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * The most of those information can be found in model.ipynb
 */
public class ModelConfig {
    public static String MODEL_FILENAME = "skin_analyzer.tflite";

    public static final int INPUT_IMG_SIZE_WIDTH = 450;
    public static final int INPUT_IMG_SIZE_HEIGHT = 450;
    public static final int FLOAT_TYPE_SIZE = 4;
    public static final int PIXEL_SIZE = 3;
    public static final int MODEL_INPUT_SIZE = FLOAT_TYPE_SIZE * INPUT_IMG_SIZE_WIDTH * INPUT_IMG_SIZE_HEIGHT * PIXEL_SIZE;

    public static final List<String> OUTPUT_LABELS = Collections.unmodifiableList(
            Arrays.asList("akiec", "bcc", "bkl", "df", "mel","nv", "vasc"));

    public static final int MAX_CLASSIFICATION_RESULTS = 7;
    public static final float CLASSIFICATION_THRESHOLD = 0.1f;
}
