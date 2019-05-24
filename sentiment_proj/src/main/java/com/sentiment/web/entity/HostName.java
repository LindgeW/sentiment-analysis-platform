package com.sentiment.web.entity;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

/**
 * Created by WuLinZhi on 2019-04-09.
 */
@Document(collection = "host")
public class HostName {
    @Id
    private Integer id;
    private String ip;
    private String port;

    public HostName() {
        this.id = 0;
        this.ip = "127.0.0.1";
        this.port = "5000";
    }

    public HostName(Integer id, String ip, String port) {
        this.id = id;
        this.ip = ip;
        this.port = port;
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getIp() {
        return ip;
    }

    public void setIp(String ip) {
        this.ip = ip;
    }

    public String getPort() {
        return port;
    }

    public void setPort(String port) {
        this.port = port;
    }

    @Override
    public String toString() {
        return ip + ":" + port;
    }
}
