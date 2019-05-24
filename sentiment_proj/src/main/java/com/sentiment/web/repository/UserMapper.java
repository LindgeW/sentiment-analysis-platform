package com.sentiment.web.repository;

import com.sentiment.web.entity.User;
import org.springframework.stereotype.Repository;

@Repository
public interface UserMapper {
    int deleteByPrimaryKey(Integer id);

    int insert(User record);

    int insertSelective(User record);

    User selectByPrimaryKey(Integer id);

    User selectByName(String username);

    int updateByPrimaryKeySelective(User record);

    int updateByPrimaryKey(User record);
}