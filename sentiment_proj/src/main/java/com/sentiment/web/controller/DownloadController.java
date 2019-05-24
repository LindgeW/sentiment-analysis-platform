package com.sentiment.web.controller;

import org.apache.commons.io.FileUtils;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import javax.servlet.http.HttpServletResponse;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.IOException;

/**
 * Created by WuLinZhi on 2019-04-19.
 */
@Controller
@RequestMapping("/user")
public class DownloadController {
    private static final String prefix = "user";
    private static final String corpusDir = "download_resources/corpus";  //资源文件夹
    private static final String word2vecDir = "download_resources/word2vec";
    private static final String lstmDir = "download_resources/lstm";

    @GetMapping("/download")
    public String download(){
        return prefix + "/download";
    }

    @GetMapping("/download_corpus")
    @ResponseBody
    public void downloadCorpus(HttpServletResponse response) throws IOException {
        File parent = new ClassPathResource(corpusDir).getFile();
        File child = new File(parent, parent.list()[0]);

        response.setContentType(MediaType.APPLICATION_OCTET_STREAM_VALUE);
        response.setHeader("Content-Disposition", "attachment; filename=" + new String(child.getName().getBytes("utf-8"),"iso-8859-1"));
        BufferedOutputStream bufOs = new BufferedOutputStream(response.getOutputStream());
        // 读取resources目录下的文件
        bufOs.write(FileUtils.readFileToByteArray(child));
        bufOs.flush();
        bufOs.close();
    }

    @GetMapping("/download_word2vec")
    @ResponseBody
    public void downloadWord2vec(HttpServletResponse response) throws IOException {
        File parent = new ClassPathResource(word2vecDir).getFile();
        File child = new File(parent, parent.list()[0]);

        response.setContentType(MediaType.APPLICATION_OCTET_STREAM_VALUE);
        response.setHeader("Content-Disposition", "attachment; filename=" + new String(child.getName().getBytes("utf-8"),"iso-8859-1"));
        BufferedOutputStream bufOs = new BufferedOutputStream(response.getOutputStream());
        // 读取resources目录下的文件
        bufOs.write(FileUtils.readFileToByteArray(child));
        bufOs.flush();
        bufOs.close();
    }

    @GetMapping("/download_lstm")
    @ResponseBody
    public void downloadLSTM(HttpServletResponse response) throws IOException {
        File parent = new ClassPathResource(lstmDir).getFile();
        File child = new File(parent, parent.list()[0]);

        response.setContentType(MediaType.APPLICATION_OCTET_STREAM_VALUE);
        response.setHeader("Content-Disposition", "attachment; filename=" + new String(child.getName().getBytes("utf-8"),"iso-8859-1"));
        BufferedOutputStream bufOs = new BufferedOutputStream(response.getOutputStream());
        // 读取resources目录下的文件
        bufOs.write(FileUtils.readFileToByteArray(child));
        bufOs.flush();
        bufOs.close();
    }
}
