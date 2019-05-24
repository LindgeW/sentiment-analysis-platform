package com.sentiment.web.controller;

import com.sentiment.web.entity.HostName;
import com.sentiment.web.service.HostService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import javax.servlet.http.HttpServletRequest;

/**
 * Created by WuLinZhi on 2019-04-22.
 */
@Controller
@RequestMapping("/user")
public class DocController {
    private final static String prefix = "user";

    @Autowired
    private HostService hostService;

    @GetMapping("/docs")
    public String docPag(HttpServletRequest request, Model model){
        HostName hostname = hostService.getHost(0);
//            HostName hostname = new HostName();
//            Cookie[] cookies = request.getCookies();
//            if(cookies != null){
//                for (Cookie cookie : cookies) {
//                    String name = cookie.getName();
//                    if(name.equalsIgnoreCase("ip")){
//                        hostname.setIp(cookie.getValue());
//                    }else if(name.equalsIgnoreCase("port")){
//                        hostname.setPort(cookie.getValue());
//                    }
//                    System.out.println(name);
//                }
//            }
            System.out.println(hostname);
            model.addAttribute("host", hostname);
            return prefix + "/documents";
    }
}
