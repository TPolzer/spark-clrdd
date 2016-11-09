package de.fau.i2;

import org.jocl.samples.JOCLSample;

import java.util.*;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;

public class SparkOCL {
	public static void main(String[] args) {
		JavaSparkContext sc = new JavaSparkContext();
		
		List<Integer> data = new ArrayList<Integer>();
		for(int i=0; i<16; ++i) {
			data.add(i);
		}
		JavaRDD<Integer> distData = sc.parallelize(data, 16);
		List<Boolean> res = distData.map(i -> JOCLSample.main(null)).collect();
		for(Boolean b : res)
			System.out.println(b);
	}
}
