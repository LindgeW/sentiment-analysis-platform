package com.sentiment.web.controller;

import com.sentiment.web.entity.RespEntity;
import com.sentiment.web.entity.RespStatus;
import com.sentiment.web.entity.User;
import org.apache.shiro.SecurityUtils;
import org.apache.shiro.authc.AuthenticationException;
import org.apache.shiro.authc.LockedAccountException;
import org.apache.shiro.authc.UnknownAccountException;
import org.apache.shiro.authc.UsernamePasswordToken;
import org.apache.shiro.subject.Subject;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.ResponseBody;

/**
 * Created by WuLinZhi on 2019-02-28.
 */
@Controller
public class LoginController {
    @GetMapping(value = {"/", "index"})
    public String index(){
        return "index"; //使用@RestController注解只能返回字符串，返回页面需使用@Controller注解
    }

    @GetMapping("/login")
    public String login(){
        return "login";
    }

    @GetMapping("/logout")
    public String logout(){
        System.out.println("已退出。。。");
        SecurityUtils.getSubject().logout();
        return "redirect:/login";
    }

    @PostMapping("/login")
    @ResponseBody
    public RespEntity ajaxLogin(User user) {
        //创建Subject实例
        Subject subject = SecurityUtils.getSubject();
        UsernamePasswordToken token = new UsernamePasswordToken(user.getUsername(), user.getPassword());
        System.out.println(user);
        String error = "";
        try {
            //将存有用户名和密码的token存进subject中
            subject.login(token);
        } catch (UnknownAccountException uae){
            error = "用户名或密码错误！";
        } catch (LockedAccountException lae){
            error = "用户已锁定或已删除！";
        } catch (AuthenticationException e){
            error = "未知错误！";
        }

        if(!subject.isAuthenticated()){ //未进行登录认证
            System.out.println("认证失败！");
            return new RespEntity(RespStatus.UNAUTHEN);
        }
        return new RespEntity<String>(RespStatus.SUCCESS, "/admin/bg");
    }
}
