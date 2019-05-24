package com.sentiment.web.utils;

import org.springframework.http.ResponseEntity;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

/**
 * Created by WuLinZhi on 2019-04-10.
 */
@Service
public class HttpClient {

    //测试网络连通性
    public Boolean test(String url){
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(600);
        RestTemplate template = new RestTemplate(requestFactory);
        //get请求
        ResponseEntity<String> responseEntity = template.getForEntity(url, String.class);
        System.out.println(responseEntity.getBody());
        return responseEntity.getStatusCodeValue() == 200;

//        String res = HttpRequestUtil.sendGet(url);
//        if(StringUtils.isBlank(res)){
//            return false;
//        }else{
//            return true;
//        }
    }

    public String client(String url, Object jsonObj){
        RestTemplate template = new RestTemplate();
//        HttpHeaders httpHeaders = new HttpHeaders();
//        httpHeaders.setContentType(MediaType.APPLICATION_JSON_UTF8);
//        httpHeaders.set(HttpHeaders.ACCEPT_CHARSET, MediaType.APPLICATION_JSON_UTF8_VALUE);
        //在URL上执行特定的HTTP方法，返回包含对象的ResponseEntity，这个对象是从响应体中映射得到的
//        HttpEntity<String, String> requestEntity = RequestEntity.post(new URI("")).contentType(MediaType.APPLICATION_JSON_UTF8).body(body);
//        ResponseEntity<String> responseEntity = template.exchange(requestEntity, String.class);
        System.out.println(url);
        System.out.println(jsonObj);
        ResponseEntity<String> responseEntity = template.postForEntity(url, jsonObj, String.class);
//        HttpEntity<Object> requestEntity = new HttpEntity<>(sendJson, httpHeaders);
//        ResponseEntity<String> responseEntity = template.postForEntity(url, requestEntity, String.class);
        System.out.println(responseEntity.getStatusCodeValue());
        System.out.println(responseEntity.getHeaders());
        System.out.println(responseEntity.getBody());

        return responseEntity.getBody();
    }
}
