package com.sentiment.web.entity;

/**
 * Created by WuLinZhi on 2019-03-03.
 */
public enum Emotion {
    POS("pos", "img/pos.png"),
    NEG("neg", "img/neg.png"),
    GEN("gen", "img/gen.png");

    private String word;
    private String url;

    Emotion(String word, String url) {
        this.word = word;
        this.url = url;
    }

    public String getWord() {
        return word;
    }

    public String getUrl() {
        return url;
    }
}
