package com.sentiment.web.entity;

/**
 * Created by WuLinZhi on 2019-03-05.
 */
public enum RespStatus {
    SUCCESS("success", 200),
    BAD("bad", 400),
    OFFLINE("offline", 444),
    UNAUTHEN("用户名或密码错误!!!", 403);
    private String statusValue;
    private Integer statusCode;

    RespStatus(String statusValue, Integer statusCode) {
        this.statusValue = statusValue;
        this.statusCode = statusCode;
    }

    public String getStatusValue() {
        return statusValue;
    }

    public Integer getStatusCode() {
        return statusCode;
    }
}
