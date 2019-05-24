package com.sentiment.web.controller;

import com.sentiment.web.entity.*;
import com.sentiment.web.service.HostService;
import com.sentiment.web.service.ParseService;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.MultipartHttpServletRequest;
import org.springframework.web.servlet.ModelAndView;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by WuLinZhi on 2019-03-01.
 */
@RestController
@RequestMapping("/user")
public class ParseController {
    private static final String prefix = "user";
    private static final Logger logger = LoggerFactory.getLogger(ParseController.class);
    private static final String charsetName = "utf-8";  //字符集编码
    private List<String> txtContents = null;  //文本输入框输入内容
    private List<String> fileContents = null;  //上传文件的内容
    private List<Remark> remarkLst = null;  //评论对象集
    private HostName hostName = null;
//    private Pager pager = null; //分页器
//    private MultipartFile mulFile = null;

    @Autowired
    private ParseService parseService;
    @Autowired
    private HostService hostService;

    @GetMapping("/parse")
    public ModelAndView toParse(HttpServletRequest request) {
        hostName = hostService.getHost(0);
        System.out.println("ip: "+hostName);

        if (hostName == null){
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
        return new ModelAndView(prefix + "/parse");
    }

    @PostMapping("/text_parse")
    public List<Remark> text_parse(@RequestParam("remarks") String remarkStr) {
        System.out.println(remarkStr);

        if(StringUtils.isBlank(remarkStr)) {
            return null;
        }
        //评论内容集
        if (txtContents == null) {
            txtContents = new ArrayList<>();
        }else {
            txtContents.clear();
        }

        for(String line : StringUtils.stripAll(StringUtils.split(remarkStr, "\n"))){
            if(StringUtils.isNoneBlank(line)){
                txtContents.add(line);
            }
        }

        remarkLst = parseService.parse(txtContents, hostName);
        return remarkLst;
    }

    @GetMapping("/file_download")
    public void download(HttpServletResponse response) throws IOException { //注：ajax访问无效
        System.out.println("downloading.....");
        String defaultFileName = "untitled.csv";
//        response.setCharacterEncoding(charsetName);
        response.setContentType(MediaType.APPLICATION_OCTET_STREAM_VALUE);
        response.setHeader("Content-Disposition", "attachment; filename=" + defaultFileName);

        BufferedOutputStream bufOs = new BufferedOutputStream(response.getOutputStream());
        for(Remark rm : remarkLst){
            bufOs.write(rm.toString().getBytes());
            bufOs.flush();
        }
        bufOs.close();
    }

//    @PostMapping("/file_upload")
//    public RespEntity upload(@RequestParam("file") MultipartFile file) throws Exception {
//        if (file==null || file.isEmpty()) {
//            return new RespEntity(RespStatus.BAD);
//        }
//        logger.info("上传成功......");
//        logger.info("文件类型：" + file.getContentType());
//        logger.info("原始文件名：" + file.getOriginalFilename());
//        logger.info("文件名：" + file.getName());
//        logger.info("文件大小(B)：" + file.getSize());
//
//        if (fileContents == null) {
//            fileContents = new ArrayList<>();
//        }else {
//            fileContents.clear();
//        }
//
//        BufferedReader br = new BufferedReader(new InputStreamReader(file.getInputStream(), charsetName));
//        String line;
//        while ((line = br.readLine()) != null) {
//            if (StringUtils.isNoneBlank(line)) {
//                fileContents.add(StringUtils.strip(line));
//            }
//        }
//        br.close();
//
//        return new RespEntity(RespStatus.SUCCESS); //注：bootstrap fileinput插件上传文件时服务端需要返回JSON格式的结果
//    }

    @PostMapping("/file_upload")
    public RespEntity upload(@RequestParam("file") MultipartFile[] files) throws Exception {
//        HttpServletRequest request
//        List<MultipartFile> files = ((MultipartHttpServletRequest)request).getFiles("file");
        if (files == null || files.length < 1) {
            return new RespEntity(RespStatus.BAD);
        }

        if (fileContents == null) {
            fileContents = new ArrayList<>();
        }else {
            fileContents.clear();
        }

        logger.info("上传成功......");
        logger.info("上传文件数量："+files.length);
        for(MultipartFile file : files){
            logger.info("文件类型：" + file.getContentType());
            logger.info("原始文件名：" + file.getOriginalFilename());
            logger.info("文件名：" + file.getName());
            logger.info("文件大小(B)：" + file.getSize());
            BufferedReader br = new BufferedReader(new InputStreamReader(file.getInputStream(), charsetName));
            String line;
            while ((line = br.readLine()) != null) {
                if (StringUtils.isNoneBlank(line)) {
                    fileContents.add(StringUtils.strip(line));
                }
            }
            br.close();
        }

        return new RespEntity(RespStatus.SUCCESS); //注：bootstrap fileinput插件上传文件时服务端需要返回JSON格式的结果
    }


    @PostMapping("/file_parse")
    public List<Remark> file_parse() {
        System.out.println("parsing....");

        if (fileContents==null || fileContents.isEmpty()){
            logger.error("上传文件为空！！！");
            return null;
        }

        remarkLst = parseService.parse(fileContents, hostName);
        return remarkLst; //bootstrap-table前端分页
    }

    @PostMapping("/word_cloud")
    public RespEntity get_keywords(@RequestParam("type") String type){
        List<KeyWord> keywordLst;
        if ("txt".equalsIgnoreCase(type)){
            if(txtContents == null || txtContents.isEmpty()){
                return new RespEntity(RespStatus.BAD);
            }

            keywordLst = parseService.getKeyWords(StringUtils.join(txtContents, ' '), hostName);
        } else if("file".equalsIgnoreCase(type)){
            if(fileContents == null || fileContents.isEmpty()){
                return new RespEntity(RespStatus.BAD);
            }

            keywordLst = parseService.getKeyWords(StringUtils.join(fileContents, ' '), hostName);
        } else{
            return new RespEntity(RespStatus.BAD);
        }

        return new RespEntity<>(RespStatus.SUCCESS, keywordLst);
    }
}

