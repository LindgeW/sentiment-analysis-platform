package com.sentiment.web.service;

import com.sentiment.web.entity.User;

/**
 * Created by WuLinZhi on 2019-04-02.
 */
public interface UserService {
    void saveUser(User user);

    User findUserByName(String username);

    void deleteUserById(Integer id);

    Boolean updatePwdByName(User user);
}
