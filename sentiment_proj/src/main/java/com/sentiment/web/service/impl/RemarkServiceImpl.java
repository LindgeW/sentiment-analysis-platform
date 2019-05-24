package com.sentiment.web.service.impl;

import com.sentiment.web.entity.Remark;
import com.sentiment.web.repository.RemarkRepository;
import com.sentiment.web.service.RemarkService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;

/**
 * Created by WuLinZhi on 2019-03-08.
 */

@Service
public class RemarkServiceImpl implements RemarkService{
    @Autowired
    private RemarkRepository remarkRepository;
//    @Autowired
//    private MongoTemplate mongoTemplate;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void save(Remark remark) {
        remarkRepository.save(remark);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void saveAll(List<Remark> remarkList) {
        remarkRepository.saveAll(remarkList);
    }

    @Override
    public List<Remark> findAll() {
        return remarkRepository.findAll();
    }

    @Override
    public void deleteById(String id) {
        remarkRepository.deleteById(id);
    }

    @Override
    public List<Remark> findByContentLike(String content) {
        return remarkRepository.findByContentLike(content);
    }

    @Override
    public List<Remark> findByContentNot(String content) {
        return remarkRepository.findByContentNot(content);
    }

    @Override
    public void deleteAll(List<Remark> rms) {
        remarkRepository.deleteAll(rms);
    }

    @Override
    public void deleteOne(Remark rm) {
        remarkRepository.delete(rm);
    }

    @Override
    public Map<String, Integer> magicCount(List<Remark> remarks) {
        HashMap<String, Integer> resultMap = new HashMap<>();
        Integer total = remarks.size();
        resultMap.put("total", total);
        Integer posNum = 0, negNum = 0, genNum = 0;
        List<String> rmLst = new ArrayList<>();
        for (Remark rm: remarks) {
            rmLst.add(rm.getContent());
            String em = rm.getEmotion();
            if (em.contains("pos")) {
                posNum ++;
            } else if (em.contains("neg")) {
                negNum ++;
            } else if (em.contains("gen")) {
                genNum ++;
            }
        }
        resultMap.put("posNum", posNum);
        resultMap.put("negNum", negNum);
        resultMap.put("genNum", genNum);

        Set<String> rmSet = new HashSet<>(rmLst);
        resultMap.put("repeat", (total - rmSet.size()));

        return resultMap;
    }
}
