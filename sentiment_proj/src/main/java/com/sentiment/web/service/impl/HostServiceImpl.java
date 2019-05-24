package com.sentiment.web.service.impl;

import com.sentiment.web.entity.HostName;
import com.sentiment.web.repository.HostRepository;
import com.sentiment.web.service.HostService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * Created by WuLinZhi on 2019-04-09.
 */

@Service
public class HostServiceImpl implements HostService {
    @Autowired
    private HostRepository hostRepository;

    /*
        insert: 若新增数据的主键已经存在，则会抛 org.springframework.dao.DuplicateKeyException 异常提示主键重复，不保存当前数据
        save: 若新增数据的主键已经存在，则会对当前已经存在的数据进行修改操作
     */
    public void saveHost(HostName hostName){
        hostRepository.save(hostName);
    }

    public HostName getHost(Integer id){
        if (hostRepository.existsById(id)){
            return hostRepository.findById(id).get();
        } else {
            return null;
        }
    }

}
