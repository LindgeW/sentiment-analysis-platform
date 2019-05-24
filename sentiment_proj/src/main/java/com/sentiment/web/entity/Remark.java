package com.sentiment.web.entity;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import javax.validation.constraints.Size;

/**
 * Created by WuLinZhi on 2019-02-28.
 */
@Document(collection = "remark")
public class Remark {
    //编号
    @Id
    String id;
    //内容
    @Size(min=8, message = "评论信息太短！！！")
    String content;
    //极性词
    String emotion;
    //表情图的url
    String imgUrl;

    public Remark() {
//        this.id = UUID.randomUUID().toString().replaceAll("-", "");
    }

    public Remark(String id, String content, String emotion, String imgUrl) {
        this.id = id;
        this.content = content;
        this.emotion = emotion;
        this.imgUrl = imgUrl;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public String getEmotion() {
        return emotion;
    }

    public void setEmotion(Emotion emotion) {
        this.emotion = emotion.getWord();
        this.imgUrl = emotion.getUrl();
    }

    public String getImgUrl() {
        return imgUrl;
    }

    public void setEmotion(String emotion) {
        this.emotion = emotion;
    }

    public void setImgUrl(String imgUrl) {
        this.imgUrl = imgUrl;
    }

    @Override
    public String toString() {
        return String.format("%s,%s\n", content,emotion);
    }
}
