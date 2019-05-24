package com.sentiment.web.controller;

import com.sentiment.web.entity.HostName;
import com.sentiment.web.entity.RespEntity;
import com.sentiment.web.entity.RespStatus;
import com.sentiment.web.service.HostService;
import com.sentiment.web.utils.HttpClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Created by WuLinZhi on 2019-04-09.
 */
@Controller
@RequestMapping("/admin")
public class HostController {
    private static final String prefix = "admin";
    @Autowired
    HostService hostService;
    @Autowired
    HttpClient httpClient;

    @GetMapping("/host")
    public String toHostPage(HttpServletRequest request, Model model){
        HostName hostName = hostService.getHost(0);
        if(hostName == null){
            hostName = new HostName();
        }

//        Cookie[] cookies = request.getCookies();
//        if(cookies != null){
//            for (Cookie cookie : cookies) {
//                String name = cookie.getName();
//                if(name.equalsIgnoreCase("ip")){
//                    hostName.setIp(cookie.getValue());
//                }else if(name.equalsIgnoreCase("port")){
//                    hostName.setPort(cookie.getValue());
//                }
//            }
//        }
        model.addAttribute("host", hostName);
        return prefix + "/hostname";
    }

    @PostMapping("/test_conn")
    @ResponseBody
    public RespEntity testConn(HostName hostname){
        String url = String.format("http://%s/to_parse", hostname);
        System.out.println(url);
        System.out.println(hostname);
        Boolean isConnected = httpClient.test(url);
        if(isConnected){
            return new RespEntity(RespStatus.SUCCESS);
        }else{
            return new RespEntity(RespStatus.OFFLINE);
        }
    }

    @PostMapping("/set_host")
    public String setHost(HostName hostName, HttpServletResponse response, Model model){
//        response.addCookie(new Cookie("ip", hostName.getIp()));
//        response.addCookie(new Cookie("port", hostName.getPort()));

        hostName.setId(0);
        hostService.saveHost(hostName);

        model.addAttribute("status", "success");
        model.addAttribute("host", hostName);
        return prefix + "/hostname";
    }
}
