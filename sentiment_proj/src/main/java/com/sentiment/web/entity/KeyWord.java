package com.sentiment.web.entity;

/**
 * Created by WuLinZhi on 2019-03-10.
 */
public class KeyWord {
    private String word;
    private Integer weight;

    public KeyWord() {
    }

    public KeyWord(String word, Integer weight) {
        this.word = word;
        this.weight = weight;
    }

    public Integer getWeight() {
        return weight;
    }

    public void setWeight(Integer weight) {
        this.weight = weight;
    }

    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }
}
