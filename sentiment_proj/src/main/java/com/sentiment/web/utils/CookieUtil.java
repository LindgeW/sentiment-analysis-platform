package com.sentiment.web.utils;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Created by WuLinZhi on 2019-04-23.
 */
@Component
public class CookieUtil {
    @Autowired
    HttpServletRequest request;

    HttpServletResponse response;


    public void setCookies(String key, String value){
        Cookie cookie = new Cookie(key, value);
//        cookie.setMaxAge(1 * 60 * 60);
        response.addCookie(cookie);
    }

    public String getCookieValue(String key){
        Cookie[] cookies = request.getCookies();
        if(cookies != null && cookies.length > 0) {
            for (Cookie cookie : cookies) {
                String name = cookie.getName();
                if(name.equalsIgnoreCase(key)){
                    return cookie.getValue();
                }
            }
        }

        return null;
    }
}
