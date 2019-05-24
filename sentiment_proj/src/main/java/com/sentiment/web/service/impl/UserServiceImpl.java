package com.sentiment.web.service.impl;

import com.sentiment.web.entity.User;
import com.sentiment.web.repository.UserMapper;
import com.sentiment.web.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * Created by WuLinZhi on 2019-04-02.
 */
@Service
public class UserServiceImpl implements UserService {
    @Autowired
    private UserMapper userMapper;

    @Override
    public void saveUser(User user) {
        userMapper.insert(user);
    }

    @Override
    public User findUserByName(String username) {
        return userMapper.selectByName(username);
    }

    @Override
    public void deleteUserById(Integer id) {
        userMapper.deleteByPrimaryKey(id);
    }

    @Override
    public Boolean updatePwdByName(User user) {
        userMapper.updateByPrimaryKeySelective(user);
        return true;
    }
}
