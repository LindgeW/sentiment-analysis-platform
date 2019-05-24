package com.sentiment.web;

import com.alibaba.fastjson.JSONObject;
import com.sentiment.web.utils.HttpRequestUtil;
import org.junit.runner.RunWith;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.stereotype.Component;
import org.springframework.test.context.junit4.SpringRunner;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by WuLinZhi on 2019-03-10.
 */
@Component
@SpringBootTest
@RunWith(SpringRunner.class)
public class Test {
    private static final String URL = "http://127.0.0.1:5000/to_parse";
    private static final String requestKey = "remarks";
    private static final String responseKey = "data";
    private static final String charsetName = "utf-8";


    @org.junit.Test
    public void parse() {
        Map<String, Object> sendMap = new HashMap<String, Object>();
        sendMap.put(requestKey, "help");
        Object sendJson = JSONObject.toJSON(sendMap);
        String acceptStr = HttpRequestUtil.sendPost(URL, sendJson, charsetName); //返回json字符串
        System.out.println("get: "+acceptStr);
        JSONObject acceptJson = JSONObject.parseObject(acceptStr);
        System.out.println(acceptJson);
        System.out.println(acceptJson.get("msg"));
        System.out.println(acceptJson.get("msg") instanceof String);
        System.out.println(acceptJson.get("code"));
        System.out.println(acceptJson.get("code") instanceof Integer);
        System.out.println(acceptJson.get("data"));
        System.out.println(acceptJson.get("data") instanceof List);


//        List<Remark> remarkLst = null;
//        if (!acceptJson.isEmpty() && acceptJson.keySet().contains(responseKey) && acceptJson.get(responseKey) instanceof List) {
//            //情感极性集
//            List<String> emotionLst = (List<String>) acceptJson.get(responseKey);
//            remarkLst = new ArrayList<Remark>();
//            for (int i = 0; i < emotionLst.size(); i++) {
//                Remark rm = new Remark();
//                rm.setId(i + 1);
//                rm.setContent(strMarks.get(i));
//                String em = emotionLst.get(i);
//                if (em.contains("pos")) {
//                    rm.setEmotion(Emotion.POS);
//                } else if (em.contains("neg")) {
//                    rm.setEmotion(Emotion.NEG);
//                } else if (em.contains("gen")) {
//                    rm.setEmotion(Emotion.GEN);
//                }
//                remarkLst.add(rm);
//            }
//        }
    }

    @org.junit.Test
    public void jsonTest() throws IOException {
        //java对象转JSON字符串
//        User user = new User();
//        user.setName("张三");
//        user.setGender("男");
//        user.setAge(20);
//        String jstr = JSONObject.toJSONString(user);
//        System.out.println(jstr);

        //JSON字符串转java对象
//        File jsonFile = ResourceUtils.getFile("classpath:jsonfiles/user.json"); //打包出错
//        File jsonFiles = ResourceUtils.getFile("classpath:jsonfiles"); //打包出错
//        File jsonFile = new ClassPathResource("jsonfiles/user.json").getFile();
//        File fileDir = new ClassPathResource("jsonfiles").getFile();
//        System.out.println(fileDir.isDirectory());
//        if(jsonFile.exists()) {
//            String jsonStr = FileUtils.readFileToString(jsonFile);
//            User user1 = JSONObject.parseObject(jsonStr, User.class);
//            System.out.println(user1);
//        }
    }
}
