// Feature definitions for each model
const modelFeatures = {
    iot23: ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
        'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
        'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS'
    ],
    toniot: ["ts", "src_port", "dst_port", "duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "dns_qclass", "dns_qtype", "dns_rcode", "http_request_body_len", "http_response_body_len", "http_status_code", "label", "src_ip_192.168.1.152", "src_ip_192.168.1.31", "src_ip_192.168.1.79", "src_ip_3.122.49.24", "dst_ip_192.168.1.152", "dst_ip_192.168.1.190", "dst_ip_192.168.1.255", "proto_tcp", "proto_udp", "service_-", "service_dns", "conn_state_OTH", "conn_state_S0", "conn_state_S1", "conn_state_S3", "conn_state_SHR", "dns_query_-", "dns_AA_-", "dns_RD_-", "dns_RA_-", "dns_rejected_-", "ssl_version_-", "ssl_cipher_-", "ssl_resumed_-", "ssl_established_-", "ssl_subject_-", "ssl_issuer_-", "http_trans_depth_-", "http_method_-", "http_uri_-", "http_version_-", "http_user_agent_-", "http_orig_mime_types_-", "http_resp_mime_types_-", "weird_name_-", "weird_name_bad_TCP_checksum", "weird_name_bad_UDP_checksum", "weird_addl_-", "weird_notice_-", "weird_notice_F"],
    unsw_nb15: ['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_sm_ips_ports'],
    cicids: ["table_id", "ip_bytes", "ip_packet", "ip_duration", "in_port", "port_bytes", "port_packet", "port_flow_count", "table_active_count", "table_lookup_count", "table_matched_count", "port_rx_packets", "port_tx_packets", "port_rx_bytes", "port_tx_bytes", "port_rx_dropped", "port_tx_dropped", "port_rx_errors", "port_tx_errors", "port_rx_frame_err", "port_rx_over_err", "port_rx_crc_err", "port_collisions", "port_duration_sec"],
    kdd:['service', 'flag', 'src_bytes', 'diff_srv_rate'],
};

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    const modelSelect = document.getElementById('modelSelect');
    const featureInputs = document.getElementById('featureInputs');
    const predictionForm = document.getElementById('predictionForm');
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    const resultsCard = document.getElementById('resultsCard');

    // Load features when model is selected
    modelSelect.addEventListener('change', function() {
        const selectedModel = this.value;
        if (selectedModel && modelFeatures[selectedModel]) {
            loadFeatureInputs(selectedModel);
            // Hide results card when model changes
            resultsCard.style.display = 'none';
        } else {
            featureInputs.innerHTML = '<div class="alert alert-info">Please select a valid model</div>';
            resultsCard.style.display = 'none';
        }
    });

    // Handle form submission
    predictionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        
        // Add model name
        const modelName = document.getElementById('modelSelect').value;
        formData.append('model_name', modelName);
        
        // Add all feature inputs manually
        const featureInputs = document.querySelectorAll('#featureInputs input');
        console.log('Found feature inputs:', featureInputs.length);
        
        featureInputs.forEach(input => {
            if (input.value !== '') {
                formData.append(input.name, input.value);
                console.log(`Adding ${input.name}: ${input.value}`);
            } else {
                formData.append(input.name, '0'); // Default to 0 if empty
                console.log(`Adding ${input.name}: 0 (default)`);
            }
        });
        
        // Debug: Log what we're sending
        console.log('FormData contents to be sent:');
        for (let [key, value] of formData.entries()) {
            console.log(`${key}: ${value}`);
        }
        
        loadingModal.show();
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            loadingModal.hide();
            displayResults(data);
            resultsCard.style.display = 'block'; // Show results card
        })
        .catch(error => {
            loadingModal.hide();
            console.error('Error:', error);
            displayError('An error occurred during prediction: ' + error.message);
            resultsCard.style.display = 'block'; // Show results card even for errors
        });
    });

   function loadFeatureInputs(modelName) {
    const features = modelFeatures[modelName];
    let html = '<h6>Feature Values:</h6><div class="row">';
    
    features.forEach((feature, index) => {
        html += `
            <div class="col-md-6 col-lg-4 mb-3">
                <div class="card h-100">
                    <div class="card-body p-3">
                        <label class="form-label small fw-bold">${feature}</label>
                        <input type="number" step="any" class="form-control form-control-sm" 
                               name="${feature}" placeholder="Enter value" value=0 required>
                    </div>
                </div>
            </div>
        `;
        
        // Close row and start new one every 3 items for better organization
        if ((index + 1) % 3 === 0 && index !== features.length - 1) {
            html += '</div><div class="row">';
        }
    });
    
    html += '</div>'; // Close the last row
    
    featureInputs.innerHTML = html;
    
    // Update feature count badge
    document.getElementById('featureCount').textContent = `${features.length} features`;
    
    // Debug: Log the loaded features
    console.log(`Loaded ${features.length} features for model: ${modelName}`);
    console.log('Features:', features);
}

    function displayResults(data) {
        const resultArea = document.getElementById('resultArea');
        
        if (data.error) {
            resultArea.innerHTML = `
                <div class="alert alert-danger">
                    <h5><i class="fas fa-exclamation-circle"></i> Error</h5>
                    <p>${data.error}</p>
                </div>
            `;
            return;
        }

        console.log('Displaying results:', data); // Debug log
        
        const predictionLower = data.prediction.toLowerCase();
        const isAttack = !predictionLower.includes('normal') && !predictionLower.includes('benign');
        const alertType = isAttack ? 'danger' : 'success';
        const icon = isAttack ? 'fa-exclamation-triangle' : 'fa-check-circle';
        const confidence = data.confidence ? (data.confidence * 100).toFixed(2) : 'N/A';
        
        let shapSection = '';
        if (data.shap_plot) {
            // Add cache busting to the image URL
            const timestamp = new Date().getTime();
            shapSection = `
                <div class="row mt-4">
                    <div class="col-12">
                        <h6><i class="fas fa-chart-bar"></i> Explanation (SHAP):</h6>
                        <div class="card">
                            <div class="card-body">
                                <p class="text-muted small">
                                    This plot shows which features most influenced the prediction. 
                                    Positive values (blue) push towards the predicted class, 
                                    while negative values (red) push away from it.
                                </p>
                                <img src="/static/${data.shap_plot}?t=${timestamp}" class="img-fluid" alt="SHAP Explanation"
                                     onerror="console.error('SHAP image failed to load:', this.src)">
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            shapSection = `
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle"></i> 
                            Feature explanation is currently unavailable. This might be due to model configuration.
                        </div>
                    </div>
                </div>
            `;
        }
        
        resultArea.innerHTML = `
            <div class="alert alert-${alertType}">
                <h4><i class="fas ${icon}"></i> ${data.prediction}</h4>
                <p class="mb-1"><strong>Confidence:</strong> ${confidence}%</p>
                ${data.is_adversarial ? '<p class="mb-0"><small><i class="fas fa-shield-alt"></i> Adversarial Simulation</small></p>' : ''}
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <h6>Input Features:</h6>
                    <div class="table-responsive">
                        <table class="table table-sm table-striped">
                            <thead>
                                <tr>
                                    <th>Feature</th>
                                    <th>Value</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${data.input_values ? Object.entries(data.input_values).map(([key, value]) => `
                                    <tr>
                                        <td><strong>${key}</strong></td>
                                        <td>${value}</td>
                                    </tr>
                                `).join('') : '<tr><td colspan="2">No feature data available</td></tr>'}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <h6>Security Recommendation:</h6>
                    <div class="alert alert-warning">
                        ${getMitigationSuggestion(data.prediction)}
                    </div>
                </div>
            </div>
            ${shapSection}
        `;
        
        // Scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth' });
    }

    function displayError(message) {
        const resultArea = document.getElementById('resultArea');
        resultArea.innerHTML = `
            <div class="alert alert-danger">
                <h5><i class="fas fa-exclamation-circle"></i> Error</h5>
                <p>${message}</p>
            </div>
        `;
    }

    function getMitigationSuggestion(attackType) {
        const suggestions = {
            'Normal': 'No action required - traffic appears normal.',
            'Benign': 'No action required - traffic appears normal.',
            'BruteForce': 'Implement account lockout policies, strengthen password requirements, use multi-factor authentication.',
            'DDoS': 'Implement rate limiting, use DDoS protection services, configure firewalls to block suspicious traffic.',
            'TCPDDOS': 'Configure TCP flood protection, implement SYN cookies, use DDoS mitigation services.',
            'UDPDDOS': 'Block unused UDP ports, implement UDP flood protection, use traffic filtering.',
            'SQLInjection': 'Use parameterized queries, input validation, web application firewalls.',
            'PortScan': 'Configure firewall rules, implement intrusion detection systems, use port knocking.',
            'CMD': 'Implement command injection protection, sanitize user inputs, use least privilege principles.',
            'Probe': 'Monitor network traffic, implement intrusion detection, block suspicious IP addresses.',
            'Samba': 'Update Samba services, restrict SMB protocols, use network segmentation.',
            'VNC': 'Use VNC over SSH tunnels, implement strong authentication, restrict VNC access to specific IPs.',
            'Analysis': 'Monitor for unusual traffic patterns, implement behavioral analysis tools.',
            'Backdoor': 'Conduct regular system scans, monitor for unusual processes, use application whitelisting.',
            'Exploits': 'Keep systems patched, use exploit mitigation techniques (ASLR, DEP), implement application security.',
            'Fuzzers': 'Implement input validation, use web application firewalls, monitor for abnormal requests.',
            'Reconnaissance': 'Limit information disclosure, monitor scan attempts, use honeypots.',
            'Shellcode': 'Implement memory protection mechanisms (ASLR, DEP), use antivirus software.',
            'Worms': 'Keep systems updated, use network segmentation, implement proper firewall rules.',
            'Dos': 'Implement rate limiting, use DoS protection services, configure load balancers.',
            'injection': 'Sanitize all user inputs, use prepared statements, implement content security policies.',
            'mitm': 'Use encrypted communications (HTTPS, VPN), implement certificate pinning, use secure protocols.',
            'password': 'Enforce strong password policies, implement multi-factor authentication, monitor for credential stuffing.',
            'ransomware': 'Maintain regular backups, use endpoint protection, educate users about phishing.',
            'scanning': 'Implement port security, use intrusion prevention systems, monitor for scan patterns.',
            'xss': 'Implement content security policy, sanitize user inputs, use XSS filters.',
            'Generic': 'General security best practices: update systems, monitor logs, implement defense in depth.'
        };

        const attackTypeLower = attackType.toLowerCase();
        for (const [key, suggestion] of Object.entries(suggestions)) {
            if (attackTypeLower.includes(key.toLowerCase())) {
                return suggestion;
            }
        }
        return suggestions['Generic'];
    }
});