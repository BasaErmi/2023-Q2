from scapy.all import *
import time


def measure_dns_query_time(dns_server, domain, count=10, rate_limit=5, period=10):
    times = []
    requests_made = 0
    start_period = time.time()

    for _ in range(count):
        current_time = time.time()

        # 检查是否需要等待以满足速率限制
        if requests_made >= rate_limit:
            time_since_start_period = current_time - start_period
            if time_since_start_period < period:
                sleep_time = period - time_since_start_period
                print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
            start_period = time.time()
            requests_made = 0

        # 创建DNS请求
        dns_request = IP(dst=dns_server) / UDP(dport=53) / DNS(rd=1, qd=DNSQR(qname=domain))

        # 记录发送时间
        start_time = time.time()

        # 发送DNS请求并等待响应
        response = sr1(dns_request, timeout=0.1, verbose=0)

        # 记录接收时间
        end_time = time.time()

        # 计算往返时间
        if response:
            round_trip_time = end_time - start_time
            times.append(round_trip_time)
        else:
            print(f"No response from {dns_server} for domain {domain}")

        requests_made += 1

    return times


def main():
    dns_server = '8.8.8.8'  # 例如，使用Google的DNS服务器
    domain = 'www.example.com'
    count = 10  # 发送10次请求

    times = measure_dns_query_time(dns_server, domain, count)

    if times:
        avg_time = sum(times) / len(times)
        print(f"Average DNS query time for {domain} to {dns_server}: {avg_time:.4f} seconds")
    else:
        print("No successful responses to calculate average time.")


if __name__ == "__main__":
    main()
