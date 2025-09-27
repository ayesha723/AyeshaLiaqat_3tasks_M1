# Ayesha Liaqat
Task 1: 
Find a website that runs on HTTP. Access this website using your device and capture network traces?
I accessed an HTTP website and captured the traffic in Wireshark.  
The screenshot below shows the captured HTTP packets from the site.  
[HTTP Traffic Screenshot](Task1_http_screenshot.png)

Task 4:  
For the HTTP based website access, answer the following after analysing collected traces of HTTP: 
1.	What is the name of website? 
The accessed website is:
www.etsy.com


2.	Find the packet that contains the first GET request for the website you have accessed. 
The first request is encapsulated inside a QUIC Initial packet. This is Packet No. 142 in my capture.
Explanation / evidence: QUIC carries the TLS handshake and then the HTTP/3 request inside encrypted QUIC streams, so the HTTP GET line is not visible in plaintext. Packet 142 is the client’s Initial QUIC packet that begins the HTTP/3 exchange.
[First GET Request](Task4_http_q2.png)


3.	Describe all headers and their values in this GET request message. 
Protocol: QUIC over UDP
Src Port: 58950 (random client port)
Dst Port: 443 (standard HTTPS/QUIC port)
Header Form: Long Header (Initial packet)
Packet Type: Initial (0) → first handshake packet
Version: QUIC version 1 (0x00000001)
Destination Connection ID: 0186fd7e29f501ae2f864e7e09f500be9a367af7
Source Connection ID: Not set (0 length here)
Token Length: 0 (no Retry token used)
Length: 1220 (payload length)
[GET Request Headers](Task4_http_q3.png)

4.	Identify the status code in the first server response.
The first response from the server returned:
HTTP/1.1 200 OK
 This means the request was successful, and the resource (webpage/text) was delivered.
 [status code ](Task4_http_q4.png)

5.	How many HTTP response messages are exchanged in total? 
By applying the filter http.response we can count the number of response packets.
In your trace, you found only one main HTTP response (200 OK).
So, the total HTTP response messages = 1.

6.	Determine whether the connection is persistent or not. Justify with evidence from packet captures. 
 In my capture, the GET request header included:
Connection: close
This means the server closed the TCP/QUIC connection after delivering the response.
Therefore, the connection is not persistent.
[non-persistent connection](Task4_http_q5&6.png)
Evidence: Presence of the Connection: close header in the GET/Response message