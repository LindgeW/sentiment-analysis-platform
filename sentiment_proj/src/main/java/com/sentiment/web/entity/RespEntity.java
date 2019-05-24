package com.sentiment.web.entity;

import java.io.Serializable;

/**
 * Created by WuLinZhi on 2019-03-05.
 */
public class RespEntity<T> implements Serializable{
    private Integer code;
    private String msg;
    private T data;

    public RespEntity(Integer code, String msg, T data) {
        this.code = code;
        this.msg = msg;
        this.data = data;
    }

    public RespEntity(RespStatus status, T data) {
        this.code = status.getStatusCode();
        this.msg = status.getStatusValue();
        this.data = data;
    }

    public RespEntity(Integer code, String msg) {
        this.code = code;
        this.msg = msg;
        data = null;
    }

    public RespEntity(RespStatus status) {
        this.code = status.getStatusCode();
        this.msg = status.getStatusValue();
        data = null;
    }

    public Integer getCode() {
        return code;
    }

    public void setCode(Integer code) {
        this.code = code;
    }

    public String getMsg() {
        return msg;
    }

    public void setMsg(String msg) {
        this.msg = msg;
    }

    public T getData() {
        return data;
    }

    public void setData(T data) {
        this.data = data;
    }

    @Override
    public String toString() {
        return "RespEntity{" +
                "code=" + code +
                ", msg='" + msg + '\'' +
                ", data=" + data +
                '}';
    }
}
