package com.sentiment.web.controller;

import com.sentiment.web.entity.RespEntity;
import com.sentiment.web.entity.RespStatus;
import com.sentiment.web.entity.User;
import com.sentiment.web.service.UserService;
import org.apache.shiro.SecurityUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

/**
 * Created by WuLinZhi on 2019-04-02.
 */
@Controller
@RequestMapping("/admin")
public class UserController {
    private final static String prefix = "admin";

    @Autowired
    private UserService userService;

    @GetMapping("/index")
    public String index(){
        return prefix + "/bg_index";
    }

    @GetMapping("/alter_pwd")
    public String toAlterPage(){
        return prefix + "/alter_pwd";
    }

    @PostMapping("/alter_pwd")
    @ResponseBody
    public RespEntity alterPwd(@RequestParam("oldPwd") String oldPwd,
                               @RequestParam("newPwd") String newPwd){
        String username = (String) SecurityUtils.getSubject().getPrincipal();
        User real = userService.findUserByName(username);
        System.out.println(real);
        if(!oldPwd.equals(real.getPassword())){
            return new RespEntity(RespStatus.UNAUTHEN);
        } else if(!oldPwd.equals(newPwd)){
            User user = new User();
            user.setId(real.getId());
//            user.setUsername(username);
            user.setPassword(newPwd);
            userService.updatePwdByName(user);
        }

        return new RespEntity(RespStatus.SUCCESS);
    }
}
