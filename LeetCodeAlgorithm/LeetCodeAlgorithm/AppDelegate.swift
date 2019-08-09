//
//  AppDelegate.swift
//  LeetCodeAlgorithm
//
//  Created by Jialin Chen on 2019/8/2.
//  Copyright © 2019年 CJL. All rights reserved.
//

import Cocoa

@NSApplicationMain
class AppDelegate: NSObject, NSApplicationDelegate {

    @IBOutlet weak var window: NSWindow!


    func applicationDidFinishLaunching(_ aNotification: Notification) {
        // Insert code here to initialize your application
        
        //swift遍历方式总结
        TraversalTest()
        
        //排序
        sortTest()
        
       //数组相关算法
        array_AlgorithmTest()
        
    }

    func applicationWillTerminate(_ aNotification: Notification) {
        // Insert code here to tear down your application
    }


}

