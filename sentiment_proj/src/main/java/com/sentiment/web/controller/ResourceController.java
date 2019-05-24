package com.sentiment.web.controller;

import com.sentiment.web.entity.RespEntity;
import com.sentiment.web.entity.RespStatus;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;

/**
 * Created by WuLinZhi on 2019-04-21.
 */
@Controller
@RequestMapping("/admin")
public class ResourceController {

    private final static String prefix = "admin";
    private final static String rootDirPath = "src/main/resources/download_resources";
    private static final Logger logger = LoggerFactory.getLogger(ResourceController.class);

    @GetMapping("/resources")
    public String resourcePage(){
        return prefix + "/resources";
    }

    @PostMapping("/upload_resources")
    @ResponseBody
    public RespEntity uploadResource(@RequestParam("file") MultipartFile uploadedFile,
                                     @RequestParam("itemId") Integer itemId) throws IOException {
        if (uploadedFile==null || uploadedFile.isEmpty()) {
            logger.error("上传文件为空！");
            return new RespEntity(RespStatus.BAD);
        }

        logger.info("上传成功......");
        logger.info("文件类型：" + uploadedFile.getContentType());
        logger.info("原始文件名：" + uploadedFile.getOriginalFilename());
        logger.info("文件名：" + uploadedFile.getName());
        logger.info("文件大小(B)：" + uploadedFile.getSize());

        String suffixDir = "";
        if(itemId == 0){
            suffixDir = "corpus";
        }else if(itemId == 1){
            suffixDir = "word2vec";
        }else if(itemId == 2){
            suffixDir = "lstm";
        }else{
            logger.error("未知项！");
            return new RespEntity(RespStatus.BAD);
        }

        File fileDir = new File(rootDirPath, suffixDir);
        if (!fileDir.exists()) {  //文件夹不存在则创建
            boolean isMk = fileDir.mkdirs();
            if (!isMk) {
                return new RespEntity(RespStatus.BAD);
            }
        }

        //清空文件夹下的所有内容
        FileUtils.cleanDirectory(fileDir);

//        保存文件
        File sourceFile = new File(fileDir.getAbsolutePath(), uploadedFile.getOriginalFilename());
        uploadedFile.transferTo(sourceFile);  //将文件保存到服务器端

        return new RespEntity(RespStatus.SUCCESS);
    }
}
