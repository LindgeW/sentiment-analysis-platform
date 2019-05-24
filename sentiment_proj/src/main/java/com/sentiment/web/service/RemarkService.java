package com.sentiment.web.service;

import com.sentiment.web.entity.Remark;

import java.util.List;
import java.util.Map;

/**
 * Created by WuLinZhi on 2019-03-08.
 */
public interface RemarkService {
    void save(Remark remark);

    void saveAll(List<Remark> remarkList);

    List<Remark> findAll();

    void deleteById(String id);

    List<Remark> findByContentLike(String content);

    List<Remark> findByContentNot(String content);

    void deleteAll(List<Remark> rms);

    void deleteOne(Remark rm);

    Map<String, Integer> magicCount(List<Remark> remarks);
}
