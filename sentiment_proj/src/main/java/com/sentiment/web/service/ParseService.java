package com.sentiment.web.service;

import com.sentiment.web.entity.HostName;
import com.sentiment.web.entity.KeyWord;
import com.sentiment.web.entity.Remark;

import java.util.List;

/**
 * Created by WuLinZhi on 2019-03-06.
 */
public interface ParseService {
    public List<Remark> parse(List<String> strMarks, HostName hostName);

    public List<KeyWord> getKeyWords(String strMarks, HostName hostName);
}
