#include <stdio.h>
#include <math.h>
#include <time.h>

int main() {
    double a, b, c, discriminant, root1, root2, realPart, imagPart;
    clock_t start, end;
    double cpu_time_used;

    printf("输入系数 a, b 和 c: ");
    scanf("%lf %lf %lf", &a, &b, &c);

    start = clock(); // 开始计时

    discriminant = b * b - 4 * a * c;

    // 计算根
    if (discriminant > 0) {
        root1 = (-b + sqrt(discriminant)) / (2 * a);
        root2 = (-b - sqrt(discriminant)) / (2 * a);
        printf("根1 = %.2lf 和 根2 = %.2lf\n", root1, root2);
    } else if (discriminant == 0) {
        root1 = root2 = -b / (2 * a);
        printf("重根 = %.2lf\n", root1);
    } else {
        realPart = -b / (2 * a);
        imagPart = sqrt(-discriminant) / (2 * a);
        printf("复数根: %.2lf+%.2lfi 和 %.2lf-%.2lfi\n", realPart, imagPart, realPart, imagPart);
    }

    end = clock(); // 结束计时
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("运行时间 = %f 秒\n", cpu_time_used);

    return 0;
}
