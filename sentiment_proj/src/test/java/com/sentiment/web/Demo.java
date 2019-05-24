package com.sentiment.web;

import org.apache.commons.lang3.StringUtils;
import org.junit.runner.RunWith;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by WuLinZhi on 2019-04-21.
 */
@SpringBootTest
@RunWith(SpringRunner.class)
public class Demo {

    @org.junit.Test
    public void listToSet(){
        List<String> lst = new ArrayList<>();
        lst.add("ab");
        lst.add("bc");
        lst.add("cd");
        lst.add("bc");
        lst.add("bc");
        lst.add("ef");
        System.out.println(lst);
        System.out.println(StringUtils.join(lst, ' '));
        System.out.println(lst.size());
        Set<String> set = new HashSet<>(lst);
        System.out.println(set);
        System.out.println(set.size());
    }

}
