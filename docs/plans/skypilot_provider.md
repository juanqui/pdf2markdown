# SkyPilot LLMProvider Implementation Plan

## Executive Summary

This document outlines a comprehensive plan for integrating SkyPilot as an LLMProvider in the pdf2markdown project. The integration will enable dynamic provisioning of GPU instances on cloud providers (particularly DataCrunch) to run open-source vision-language models for PDF processing. This approach provides cost-effective scaling, automatic resource management, and support for state-of-the-art models like Qwen2.5-VL.

## Requirements Analysis

### Functional Requirements

1. **LLMProvider Interface Compliance**
   - Implement all abstract methods from `LLMProvider` base class
   - Support `invoke_with_image`, `invoke_with_image_base64`, and `invoke` methods
   - Return proper `LLMResponse` objects with content and metadata

2. **Dynamic Instance Provisioning**
   - Provision GPU instances on-demand via SkyPilot
   - Support multiple cloud providers with DataCrunch as primary target
   - Handle spot and on-demand instance types
   - Automatic instance teardown after processing

3. **Model Support**
   - Primary: Qwen2.5-VL (7B/14B/32B variants)
   - Secondary: LLaVA, CogVLM, InternVL
   - Support for custom model endpoints via SGLang/vLLM

4. **Lifecycle Management**
   - Lazy provisioning (only when first request arrives)
   - Keep-alive periods for batch processing
   - Graceful shutdown with resource cleanup
   - Connection pooling and reuse

5. **Cost Optimization**
   - Spot instance preference with fallback
   - Auto-shutdown after idle timeout
   - Instance right-sizing based on model
   - Cost tracking and budget limits

### Non-Functional Requirements

1. **Performance**
   - Cold start: < 5 minutes for instance provisioning
   - Warm start: < 100ms for subsequent requests
   - Support concurrent page processing
   - Connection pooling for efficiency

2. **Reliability**
   - Automatic retry on provisioning failures
   - Regional failover support
   - Health checking before processing
   - Graceful degradation to fallback providers

3. **Security**
   - SSH key-based authentication only
   - Encrypted communication channels
   - No hardcoded credentials
   - Temporary resource cleanup

4. **Observability**
   - Detailed logging of provisioning steps
   - Cost tracking per session
   - Performance metrics collection
   - Error tracking with context

## Proposed Architecture

### High-Level Design

```
┌────────────────────────────────────────────────────────┐
│                 pdf2markdown Application            │
├────────────────────────────────────────────────────────┤
│                    Pipeline System                      │
│  ┌─────────────────────────────────────────────────┐  │
│  │              SimpleLLMPageParser                 │  │
│  │                      │                           │  │
│  │         ┌────────────┴────────────┐             │  │
│  │         │    LLMProvider Factory   │             │  │
│  │         └────────────┬────────────┘             │  │
│  │                      │                           │  │
│  │   ┌──────────────────┴──────────────────┐      │  │
│  │   │         SkyPilotLLMProvider         │      │  │
│  │   └──────────────────┬──────────────────┘      │  │
│  └─────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────┤
│                  SkyPilot Integration Layer            │
│  ┌──────────────────────────────────────────────────┐ │
│  │         SkyPilotClusterManager                   │ │
│  │  ┌──────────────┐  ┌──────────────┐            │ │
│  │  │ Provisioner  │  │ Health Check  │            │ │
│  │  └──────────────┘  └──────────────┘            │ │
│  │  ┌──────────────┐  ┌──────────────┐            │ │
│  │  │ Model Deploy │  │ Cost Monitor │            │ │
│  │  └──────────────┘  └──────────────┘            │ │
│  └──────────────────────────────────────────────────┘ │
├────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                │
│  ┌──────────────────────────────────────────────────┐ │
│  │              SkyPilot Orchestrator               │ │
│  │                      │                           │ │
│  │         ┌────────────┴────────────┐             │ │
│  │         │    Cloud Provider API    │             │ │
│  │         │      (DataCrunch)        │             │ │
│  │         └──────────────────────────┘             │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. SkyPilotLLMProvider
- **Responsibility**: Implements LLMProvider interface for SkyPilot-managed models
- **Key Methods**:
  - `__init__`: Initialize with configuration
  - `invoke_with_image`: Process image with remote model
  - `invoke_with_image_base64`: Process base64 image
  - `invoke`: Text-only processing
  - `cleanup`: Teardown resources
  - `validate_config`: Configuration validation

#### 2. SkyPilotClusterManager
- **Responsibility**: Manages cluster lifecycle and model deployment
- **Key Methods**:
  - `provision_cluster`: Create GPU instance
  - `deploy_model`: Install and start model server
  - `health_check`: Verify cluster and model status
  - `teardown_cluster`: Clean up resources
  - `get_endpoint`: Return model API endpoint

#### 3. ModelEndpointClient
- **Responsibility**: Communicate with deployed model endpoints
- **Key Methods**:
  - `call_model`: Send requests to model
  - `prepare_payload`: Format request data
  - `parse_response`: Extract model output
  - `handle_errors`: Process API errors

#### 4. CostMonitor
- **Responsibility**: Track and control cloud spending
- **Key Methods**:
  - `track_usage`: Monitor instance runtime
  - `calculate_cost`: Compute current costs
  - `check_budget`: Verify budget limits
  - `generate_report`: Create cost reports

## Technology Choices

### Stack Selection

1. **SkyPilot**: Cloud orchestration framework
   - **Justification**: Multi-cloud support, built-in optimization, active development
   - **Version**: Latest stable (0.7.x)

2. **SGLang**: Model serving framework
   - **Justification**: Optimized for VLMs, better performance than raw transformers
   - **Alternative**: vLLM for text-only models

3. **DataCrunch**: Primary cloud provider
   - **Justification**: Cost-effective GPUs, good availability, API support
   - **Fallback**: AWS, GCP via SkyPilot abstraction

4. **Qwen2.5-VL**: Primary vision-language model
   - **Justification**: State-of-the-art performance, multiple size options
   - **Alternatives**: LLaVA-1.6, CogVLM2

5. **HTTPX**: Async HTTP client
   - **Justification**: Better async support than requests, connection pooling
   - **Usage**: Communication with model endpoints

### Configuration Schema

```python
@dataclass
class SkyPilotProviderConfig:
    # Basic settings
    provider_type: str = "skypilot"
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # Instance configuration
    cloud_provider: str = "datacrunch"
    instance_type: str = "V100"
    use_spot: bool = True
    disk_size: int = 50  # GB
    
    # Model serving configuration
    serving_framework: str = "sglang"  # or "vllm"
    serving_port: int = 30000
    tensor_parallel_size: int = 1
    
    # Resource management
    idle_timeout: int = 300  # seconds
    max_idle_time: int = 900  # seconds before teardown
    keep_warm: bool = False  # Keep instance running
    
    # Cost control
    max_hourly_cost: float = 10.0
    budget_limit: float = 100.0
    
    # Advanced options
    regions: List[str] = ["FIN-01"]
    retry_count: int = 3
    health_check_timeout: int = 300
    setup_script: Optional[str] = None
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

#### Milestone 1.1: Basic SkyPilotLLMProvider
- [ ] Create `src/pdf2markdown/llm_providers/skypilot.py`
- [ ] Implement basic LLMProvider interface
- [ ] Add configuration schema to `config/schemas.py`
- [ ] Update factory pattern in `llm_providers/factory.py`

#### Milestone 1.2: Cluster Management
- [ ] Create `src/pdf2markdown/llm_providers/skypilot_cluster.py`
- [ ] Implement cluster provisioning logic
- [ ] Add health checking mechanisms
- [ ] Implement graceful teardown

#### Milestone 1.3: Model Deployment
- [ ] Create SkyPilot task YAML templates
- [ ] Implement model deployment scripts
- [ ] Add SGLang server configuration
- [ ] Test model endpoint connectivity

### Phase 2: Model Integration (Week 2-3)

#### Milestone 2.1: Endpoint Communication
- [ ] Create `ModelEndpointClient` class
- [ ] Implement request/response handling
- [ ] Add retry logic with backoff
- [ ] Handle connection pooling

#### Milestone 2.2: Image Processing
- [ ] Implement image encoding/preparation
- [ ] Add prompt formatting for VLMs
- [ ] Handle response parsing
- [ ] Add error recovery

#### Milestone 2.3: Multi-Model Support
- [ ] Add Qwen2.5-VL configuration
- [ ] Support multiple model sizes
- [ ] Add LLaVA as alternative
- [ ] Implement model selection logic

### Phase 3: Resource Management (Week 3-4)

#### Milestone 3.1: Lifecycle Management
- [ ] Implement lazy provisioning
- [ ] Add idle timeout handling
- [ ] Create keep-alive mechanism
- [ ] Handle graceful shutdown

#### Milestone 3.2: Cost Optimization
- [ ] Implement cost tracking
- [ ] Add budget enforcement
- [ ] Support spot instances
- [ ] Add instance right-sizing

#### Milestone 3.3: Reliability Features
- [ ] Add regional failover
- [ ] Implement circuit breaker
- [ ] Add health monitoring
- [ ] Create fallback mechanisms

### Phase 4: Integration & Testing (Week 4-5)

#### Milestone 4.1: Pipeline Integration
- [ ] Test with existing pipeline
- [ ] Validate async operations
- [ ] Ensure proper cleanup
- [ ] Performance benchmarking

#### Milestone 4.2: Configuration Management
- [ ] Add CLI parameters
- [ ] Update configuration files
- [ ] Add environment variables
- [ ] Document configuration

#### Milestone 4.3: Testing Suite
- [ ] Unit tests for provider
- [ ] Integration tests
- [ ] End-to-end tests
- [ ] Performance tests

### Phase 5: Production Readiness (Week 5-6)

#### Milestone 5.1: Monitoring & Logging
- [ ] Add structured logging
- [ ] Implement metrics collection
- [ ] Create dashboards
- [ ] Add alerting

#### Milestone 5.2: Documentation
- [ ] API documentation
- [ ] Usage examples
- [ ] Configuration guide
- [ ] Troubleshooting guide

#### Milestone 5.3: Optimization
- [ ] Performance tuning
- [ ] Resource optimization
- [ ] Cost analysis
- [ ] Load testing

## Risk Assessment

### Technical Risks

1. **Cold Start Latency**
   - **Risk**: 5-minute provisioning time may be too long
   - **Mitigation**: Implement warm pool, use faster instance types
   - **Fallback**: Keep instances warm for known workloads

2. **Spot Instance Interruption**
   - **Risk**: Processing interrupted mid-document
   - **Mitigation**: Implement checkpointing, use on-demand fallback
   - **Impact**: Medium - requires retry logic

3. **Model Memory Requirements**
   - **Risk**: Large models may not fit on smaller GPUs
   - **Mitigation**: Dynamic instance selection, model quantization
   - **Impact**: High - affects cost and availability

4. **Network Latency**
   - **Risk**: High latency to remote endpoints
   - **Mitigation**: Regional deployment, connection pooling
   - **Impact**: Low-Medium - affects throughput

### Operational Risks

1. **Cost Overruns**
   - **Risk**: Unexpected cloud costs
   - **Mitigation**: Budget limits, monitoring, alerts
   - **Severity**: High - requires immediate attention

2. **Provider Outages**
   - **Risk**: DataCrunch availability issues
   - **Mitigation**: Multi-provider support via SkyPilot
   - **Severity**: Medium - have fallback providers

3. **Dependency Updates**
   - **Risk**: Breaking changes in SkyPilot/SGLang
   - **Mitigation**: Pin versions, comprehensive testing
   - **Severity**: Low - manageable with testing

## Testing Strategy

### Unit Testing

```python
class TestSkyPilotLLMProvider:
    def test_initialization(self):
        # Test provider initialization
        
    def test_config_validation(self):
        # Test configuration validation
        
    def test_invoke_methods(self):
        # Test all invoke methods
        
    def test_cleanup(self):
        # Test resource cleanup
```

### Integration Testing

```python
class TestSkyPilotIntegration:
    def test_cluster_provisioning(self):
        # Test actual cluster creation
        
    def test_model_deployment(self):
        # Test model server deployment
        
    def test_end_to_end_processing(self):
        # Test complete PDF processing
        
    def test_failover(self):
        # Test failure scenarios
```

### Performance Testing

- Measure cold start times
- Benchmark throughput (pages/minute)
- Test concurrent processing
- Monitor resource utilization
- Validate cost tracking

### Load Testing

- Simulate high-volume processing
- Test queue management
- Verify scaling behavior
- Monitor system stability

## Success Metrics

### Performance Metrics
- **Cold Start Time**: < 5 minutes
- **Warm Request Latency**: < 100ms
- **Throughput**: > 10 pages/minute
- **Availability**: > 99%

### Cost Metrics
- **Cost per Page**: < $0.10
- **Spot Instance Usage**: > 70%
- **Idle Time**: < 20%
- **Budget Accuracy**: ± 5%

### Quality Metrics
- **Model Accuracy**: Comparable to GPT-4V
- **Error Rate**: < 1%
- **Retry Success Rate**: > 95%
- **User Satisfaction**: > 90%

### Operational Metrics
- **Deployment Time**: < 30 minutes
- **Recovery Time**: < 5 minutes
- **Monitoring Coverage**: 100%
- **Documentation Completeness**: 100%

## Security Considerations

### Authentication & Authorization
- SSH key management via SkyPilot
- API key rotation for model endpoints
- IAM roles for cloud providers
- Least privilege access

### Data Security
- Encrypted data in transit (TLS)
- Temporary file cleanup
- No persistent storage of sensitive data
- Secure credential storage

### Network Security
- Private VPC deployment option
- Security group configuration
- Firewall rules for endpoints
- DDoS protection

### Compliance
- GDPR compliance for EU data
- Data residency requirements
- Audit logging
- Security scanning

## Maintenance & Operations

### Monitoring Strategy
1. **Application Metrics**
   - Request count and latency
   - Error rates and types
   - Model performance metrics
   - Queue depths

2. **Infrastructure Metrics**
   - Instance utilization
   - Network throughput
   - Disk usage
   - Memory consumption

3. **Cost Metrics**
   - Hourly/daily spend
   - Cost per request
   - Budget utilization
   - Spot savings

### Alerting Rules
- High error rate (> 5%)
- Budget threshold (80%, 90%, 100%)
- Instance failures
- Model server crashes
- Queue backlog

### Runbook Procedures
1. **Instance Provisioning Failure**
   - Check cloud provider status
   - Verify credentials
   - Try alternate regions
   - Fall back to on-demand

2. **Model Server Crash**
   - Check logs for errors
   - Verify memory availability
   - Restart server
   - Re-deploy if needed

3. **Cost Overrun**
   - Immediate instance termination
   - Investigate usage patterns
   - Adjust limits
   - Review spot pricing

## Migration Strategy

### Gradual Rollout
1. **Phase 1**: Internal testing with small documents
2. **Phase 2**: Beta users with opt-in flag
3. **Phase 3**: Default for specific use cases
4. **Phase 4**: General availability

### Backward Compatibility
- Maintain existing LLMProvider interface
- Support configuration migration
- Provide fallback to OpenAI
- Preserve existing behavior

### Data Migration
- No data migration required
- Configuration updates only
- Optional performance history import
- Cost data tracking from day one

## Future Enhancements

### Short-term (3 months)
1. **Additional Models**
   - InternVL support
   - Gemini via VertexAI
   - Custom fine-tuned models

2. **Performance Optimizations**
   - Batch processing optimization
   - Caching layer
   - Prefetching mechanisms

3. **Cost Features**
   - Detailed cost breakdown
   - Cost prediction
   - Optimization recommendations

### Medium-term (6 months)
1. **Advanced Features**
   - Multi-region deployment
   - Auto-scaling
   - A/B testing framework

2. **Integration Expansion**
   - Kubernetes deployment
   - Docker containerization
   - CI/CD pipeline integration

3. **Monitoring Enhancement**
   - Custom dashboards
   - Anomaly detection
   - Predictive maintenance

### Long-term (12 months)
1. **Platform Features**
   - Model marketplace
   - Custom model training
   - Federated learning

2. **Enterprise Features**
   - Private cloud support
   - Compliance certifications
   - SLA guarantees

3. **Ecosystem Integration**
   - Plugin architecture
   - Third-party integrations
   - API marketplace

## Conclusion

The SkyPilot LLMProvider implementation represents a significant enhancement to the pdf2markdown project, enabling cost-effective access to state-of-the-art vision-language models through dynamic cloud provisioning. The phased implementation approach ensures manageable risk while delivering value incrementally.

Key success factors include:
- Robust error handling and recovery mechanisms
- Comprehensive testing at all levels
- Clear documentation and examples
- Active monitoring and cost control
- Gradual rollout with fallback options

The implementation will provide users with:
- Access to powerful open-source models
- Significant cost savings (up to 70% with spot instances)
- Automatic resource management
- Seamless integration with existing workflows
- Flexibility to choose models and providers

With proper execution of this plan, the SkyPilot integration will position pdf2markdown as a leading solution for cost-effective, scalable document processing using cutting-edge AI models.