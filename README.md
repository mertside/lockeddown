# LockedDown
LockedDown: Exploiting Contention on Host-GPU PCIe Bus for Fun and Profit

# Abstract
The deployment of modern graphics processing units (GPUs) has grown rapidly in both traditional and cloud computing. 
Nevertheless, the potential security issues brought forward by this extensive deployment have not been thoroughly investigated. 
In this paper, we disclose a new exploitable side-channel vulnerability that ubiquitously exists in systems equipped with modern GPUs. 
This vulnerability is due to measurable contention caused on the host-GPU PCIe bus. 
To demonstrate the exploitability of this vulnerability, we conduct two case studies. 
In the first case study, we exploit the vulnerability to build a cross-VM covert channel that works on virtualized NVIDIA GPUs. 
To the best of our knowledge, this is the first work that explores covert channel attacks under the circumstances of virtualized GPUs. 
The covert channel can reach a speed up to 90 kbps with a considerably low error rate. 
In the second case study, we exploit the vulnerability to mount a website fingerprint- ing attack that can accurately infer which web pages are browsed by a user. 
The attack is evaluated against popular browsers like Chrome and Firefox on both Windows and Linux, and the results show that this fingerprinting method can achieve up to 95.2% accuracy. 
In addition, the attack is evaluated against Tor browser, and up to 90.6% accuracy can be achieved.

# Technical Support
If you need assistance, you can contact the developer at: mert (dot) side (at) ttu (dot) edu