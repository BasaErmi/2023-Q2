#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <vector>

using namespace std;

const int INF = 2147483647;
const int SIZE = 4096;
const string fileName = "updated_mouse.csv";
const string testFileName = "test_data2.txt";

// 从CSV文件中读取图数据
void readData(vector<vector<double>>& graph) {
    ifstream csvData(fileName);
    if (!csvData) {
        cerr << "打开文件失败" << endl;
        exit(1);
    }

    string line;
    getline(csvData, line); // 跳过CSV文件的第一行

    // 读取每一行数据
    while (getline(csvData, line)) {
        istringstream sin(line);
        vector<string> tokens;
        string token;

        // 按照逗号分隔符解析每一行
        while (getline(sin, token, ',')) {
            tokens.push_back(token);
        }

        // 将解析后的数据存入图的邻接矩阵中
        int pointA = stoi(tokens[0]);
        int pointB = stoi(tokens[1]);
        double distance = stod(tokens[2]);

        graph[pointA][pointB] = distance;
        graph[pointB][pointA] = distance;
    }
}

// Floyd算法并行实现
void floyd(vector<vector<double>>& graph, vector<vector<double>>& next, int threadNum) {
    #pragma omp parallel for num_threads(threadNum)
    for (int k = 0; k < SIZE; k++) {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (graph[i][k] != INF && graph[k][j] != INF && graph[i][k] + graph[k][j] < graph[i][j]) {
                    graph[i][j] = graph[i][k] + graph[k][j];
                    next[i][j] = k;
                }
            }
        }
    }
}

// 输出结果
void Output(const vector<vector<double>>& graph, const vector<vector<double>>& next) {
    ifstream testData(testFileName);
    if (!testData) {
        cerr << "打开文件失败" << endl;
        exit(1);
    }

    string line;
    while (getline(testData, line)) {
        istringstream sin(line);
        vector<string> tokens;
        string token;

        // 按照空格分隔符解析每一行
        while (getline(sin, token, ' ')) {
            tokens.push_back(token);
        }

        // 获取起点和终点
        int pointA = stoi(tokens[0]);
        int pointB = stoi(tokens[1]);

        ofstream output("result2.txt", ios::app);
        if (!output) {
            cerr << "打开文件失败"  << endl;
            exit(1);
        }

        // 输出最短路径距离和路径
        string result = (graph[pointA][pointB] == INF) ? "INF" : to_string(graph[pointA][pointB]);
        output << pointA << " " << pointB << " " << result << " ";

        if (graph[pointA][pointB] != INF) {
            output << "Path: " << pointA << "->";
            int k = next[pointA][pointB];
            while (k != -1) {
                output << k << "->";
                k = next[k][pointB];
            }
            output << pointB << endl;
        }
    }
}


int main() {
    // 初始化图的邻接矩阵和路径矩阵
    vector<vector<double>> graph(SIZE, vector<double>(SIZE, INF));
    vector<vector<double>> next(SIZE, vector<double>(SIZE, -1));

    for (int i = 0; i < SIZE; i++) {
        graph[i][i] = 0;
    }

    // 读取数据
    readData(graph);

    // 执行floyd算法并输出运行时间
    for (int i = 1; i <= 16; i *= 2) {
        double start = omp_get_wtime();
        floyd(graph, next, i);
        double end = omp_get_wtime();
        cout << "线程数: " << i << " 时间: " << end - start << endl;
    }

    // 输出结果
    Output(graph, next);

    cout << "程序运行完毕" << endl;

    return 0;
}
