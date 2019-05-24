package com.sentiment.web.service;

import com.sentiment.web.entity.HostName;

/**
 * Created by WuLinZhi on 2019-04-09.
 */
public interface HostService {
    void saveHost(HostName hostName);

    HostName getHost(Integer id);
}
