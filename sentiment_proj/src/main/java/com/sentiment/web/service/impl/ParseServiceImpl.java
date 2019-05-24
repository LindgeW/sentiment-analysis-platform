package com.sentiment.web.service.impl;

import com.alibaba.fastjson.JSONObject;
import com.sentiment.web.entity.Emotion;
import com.sentiment.web.entity.HostName;
import com.sentiment.web.entity.KeyWord;
import com.sentiment.web.entity.Remark;
import com.sentiment.web.repository.RemarkRepository;
import com.sentiment.web.service.ParseService;
import com.sentiment.web.utils.HttpClient;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.nio.charset.Charset;
import java.util.*;

/**
 * Created by WuLinZhi on 2019-03-06.
 */
@Service
public class ParseServiceImpl implements ParseService {
    private static final String requestKey = "remarks";
    private static final String responseKey = "data";
    private static final String charsetName = "utf-8";

    @Autowired
    HttpClient httpClient;
    @Autowired
    RemarkRepository remarkRepository;

    private String getURL(HostName hostName, String path){
        String protocol = "http";
        return String.format("%s://%s/%s", protocol, hostName, path);
    }

    public List<Remark> parse(List<String> strMarks, HostName hostName) {
        String url = getURL(hostName, "to_parse");

        Map<String, Object> sendMap = new HashMap<String, Object>();
        sendMap.put(requestKey, strMarks);
        Object sendJson = JSONObject.toJSON(sendMap);
        String acceptStr = httpClient.client(url, sendJson);
//        String acceptStr = HttpRequestUtil.sendPost(url, sendJson, charsetName); //返回json字符串
        JSONObject acceptJson = JSONObject.parseObject(acceptStr);
        Integer code = (Integer) acceptJson.get("code");
        List<Remark> remarkLst = null;
        if(code == 200){
            if (acceptJson.keySet().contains(responseKey) && acceptJson.get(responseKey) instanceof List) {
                //情感极性集
                List<String> emotionLst = (List<String>) acceptJson.get(responseKey);
                remarkLst = new ArrayList<Remark>();
                for (int i = 0; i < emotionLst.size(); i++) {
                    Remark rm = new Remark();
                    rm.setId(UUID.randomUUID().toString().replaceAll("-", ""));
                    rm.setContent(StringUtils.toEncodedString(strMarks.get(i).getBytes(), Charset.forName("UTF-8")));
                    String em = emotionLst.get(i);
                    if (em.contains("pos")) {
                        rm.setEmotion(Emotion.POS);
                    } else if (em.contains("neg")) {
                        rm.setEmotion(Emotion.NEG);
                    } else if (em.contains("gen")) {
                        rm.setEmotion(Emotion.GEN);
                    }
                    remarkLst.add(rm);
                }

                remarkRepository.saveAll(remarkLst);
            }
        }
        return remarkLst;
    }

    public List<KeyWord> getKeyWords(String strMarks, HostName hostName) {
        String url = getURL(hostName, "wordcloud");

        Map<String, Object> sendMap = new HashMap<String, Object>();
        sendMap.put(requestKey, strMarks);
        Object sendJson = JSONObject.toJSON(sendMap);
        System.out.println(url);
        String acceptStr = httpClient.client(url, sendJson);
//        String acceptStr = HttpRequestUtil.sendPost(url, sendJson, charsetName); //返回json字符串
        JSONObject acceptJson = JSONObject.parseObject(acceptStr);
        Integer code = (Integer) acceptJson.get("code");
        List<KeyWord> keywordLst = null;
        if(code == 200){
            if (acceptJson.keySet().contains(responseKey) && acceptJson.get(responseKey) instanceof List) {
                //情感极性集
                List<String> keywords = (List<String>) acceptJson.get(responseKey);
                keywordLst = new ArrayList<KeyWord>();
                for(String kwStr : keywords){
                    JSONObject jobj = JSONObject.parseObject(kwStr);
                    String name = (String)jobj.get("name");
                    Integer value = (Integer)jobj.get("value");
                    KeyWord kwBean = new KeyWord(StringUtils.toEncodedString(name.getBytes(), Charset.forName("UTF-8")), value);
                    keywordLst.add(kwBean);
                }
            }
        }
        return keywordLst;
    }

}
