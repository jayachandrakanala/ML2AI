# Large-Scale E-Commerce Marketplace Architecture

> A system-design study of how a large e-commerce marketplace can serve buyers and sellers, manage catalog and inventory, coordinate fulfillment, and handle high request volumes. The technologies below are representative options and examples; they are not claims about Amazon's exact internal implementation.

## Table of Contents

1. [Architecture Scope and Design Goals](#1-architecture-scope-and-design-goals)
2. [End-to-End Request Lifecycle](#2-end-to-end-request-lifecycle)
3. [Marketplace Domains and Data Ownership](#3-marketplace-domains-and-data-ownership)
4. [Buyer and Seller Workflows](#4-buyer-and-seller-workflows)
5. [Catalog, Offers, and Search](#5-catalog-offers-and-search)
6. [Pricing and Buy Box Evaluation](#6-pricing-and-buy-box-evaluation)
7. [Cart, Checkout, and Inventory Reservation](#7-cart-checkout-and-inventory-reservation)
8. [Supply Chain, Fulfillment, and Delivery](#8-supply-chain-fulfillment-and-delivery)
9. [Personalization and Recommendations](#9-personalization-and-recommendations)
10. [Caching and Content Delivery](#10-caching-and-content-delivery)
11. [Data Stores and Distributed Data Patterns](#11-data-stores-and-distributed-data-patterns)
12. [Resilience, Security, and Observability](#12-resilience-security-and-observability)
13. [Redis Vector Search and Java WebFlux Deep Dive](#13-redis-vector-search-and-java-webflux-deep-dive)
14. [Serving Millions of Concurrent Users](#14-serving-millions-of-concurrent-users)
15. [Interview Discussion Checklist](#15-interview-discussion-checklist)
16. [Further Reading](#16-further-reading)

---

## 1. Architecture Scope and Design Goals

A large marketplace brings together buyers, third-party sellers, products, offers, inventory, payments, warehouses, carriers, and customer support. The platform must coordinate these domains while remaining responsive during demand spikes and infrastructure failures.

Key design goals include:

- **Availability:** Keep browsing and essential purchase flows operating through partial failures.
- **Low latency:** Serve common reads from nearby edge locations, caches, and appropriately indexed stores.
- **Independent scaling:** Scale catalog, search, checkout, and fulfillment according to their different traffic profiles.
- **Correctness:** Protect financial transactions and inventory from duplicate or oversold operations.
- **Resilience:** Use timeouts, retries, backpressure, isolation, and graceful degradation.
- **Consistency appropriate to the workflow:** Stronger consistency for critical state transitions; eventual consistency for derived views and search indexes where acceptable.
- **Security and privacy:** Authenticate users and services, protect payment data, limit access, and govern personal data.
- **Operational visibility:** Trace requests and asynchronous workflows across service boundaries.

The scale figures sometimes used in marketplace discussions (for example, millions of sellers or billions of items) are scenario assumptions, not measurements asserted by this document.

---

## 2. End-to-End Request Lifecycle

A buyer request travels through multiple layers. This example follows a product-page request; checkout has additional authorization and consistency requirements.

```text
Buyer device
    |
    v
DNS and edge routing
    |
    +--> CDN cache hit: return cacheable static content
    |
    v
WAF / DDoS controls
    |
    v
API gateway or reverse proxy
    |
    v
Load balancer
    |
    v
Application services
    |
    +--> Cache
    +--> Catalog / offer / inventory services
    +--> Search or recommendation services
    |
    v
Databases and asynchronous event processing
```

### 2.1 Edge and network layer

1. **DNS and routing:** DNS resolves the service name. Anycast or provider routing may direct traffic toward a suitable region or edge location.
2. **Content delivery network:** A CDN such as CloudFront or Akamai can cache static assets, including images, stylesheets, scripts, and selected cacheable fragments.
3. **Transport security:** TLS protects communication between clients and service endpoints. TLS termination may happen at the edge, a gateway, or another controlled point in the network.

The CDN should not cache personalized or sensitive responses unless the cache key and privacy behavior are explicitly designed for that purpose.

### 2.2 Security and request routing

- **Web application firewall (WAF):** Applies request filters and rules for common web threats and abusive traffic.
- **DDoS protection:** Absorbs or mitigates volumetric and protocol-level attacks at the network edge.
- **API gateway or reverse proxy:** Provides an entry point for authentication enforcement, request validation, rate limiting, routing, and API versioning.
- **Service discovery or service mesh:** Helps internal services locate one another and can standardize transport security, telemetry, and traffic policies.

Representative technologies include AWS WAF and Shield, NGINX, Envoy, API Gateway, Istio, or Linkerd. The choice depends on deployment and operational needs.

### 2.3 Load balancing and application services

Layer 4 load balancers distribute TCP or UDP connections. Layer 7 load balancers can route using HTTP properties such as host, path, or headers. Health checks remove unhealthy targets from rotation.

Application services may run on virtual machines or containers orchestrated by systems such as Kubernetes, Amazon ECS, or managed container platforms. Horizontal scaling adds service instances as demand changes, subject to startup time, capacity limits, and downstream dependencies.

### 2.4 Cache, services, and persistence

Application services can use in-memory caches for frequently read data, then query the system of record on a cache miss. Data is persisted in stores selected for the access pattern: for example, relational databases for transactions, key-value or document stores for high-volume lookups, and search indexes for text retrieval.

---

## 3. Marketplace Domains and Data Ownership

Domain decomposition helps teams evolve business capabilities independently. A bounded context should own its data and expose well-defined APIs or events rather than allowing other services to write directly into its database.

```text
                       API gateway / service entry
                                  |
       +------------------+-----+-------+------------------+
       |                  |             |                  |
       v                  v             v                  v
    Catalog            Offers        Orders             Seller
       |                  |             |                  |
       +------------------+-------------+------------------+
                                  |
                   Events and asynchronous workflows
                                  |
       +------------------+-----+-------+------------------+
       |                  |             |                  |
       v                  v             v                  v
   Inventory          Payments       Fulfillment         Search
```

### 3.1 Core bounded contexts

| Domain | Responsibilities | Example data |
|---|---|---|
| Buyer identity | Accounts, preferences, addresses, authorization | Buyer profile, saved addresses |
| Seller management | Seller onboarding, permissions, policies, performance | Seller account, eligibility |
| Catalog / product information | Product definitions, attributes, variations, media | Product, category, parent-child variation |
| Offers and pricing | Seller-specific offer, price, shipping terms, availability | Offer, price, fulfillment method |
| Inventory | Available, reserved, and inbound quantities | Stock by item and fulfillment location |
| Cart | Buyer-selected items and quantities | Cart document, item selections |
| Order | Purchase intent and order lifecycle | Order, line item, state transition |
| Payment | Authorization, capture, refund, settlement | Payment intent, transaction references |
| Fulfillment | Picking, packing, shipment creation | Fulfillment order, package |
| Delivery | Carrier handoff, tracking, delivery confirmation | Route, shipment event |
| Search and discovery | Query indexing, filters, relevance | Search document, facets |
| Recommendations | Candidate generation, ranking, personalization | Features, embeddings, recommendations |

Services should define clear ownership and contracts. A microservice boundary is not valuable by itself; it should reduce coupling or enable independent scaling, deployment, or team ownership.

### 3.2 CQRS and projections

Command Query Responsibility Segregation (CQRS) separates write operations from read-optimized representations:

1. A command updates the authoritative domain state.
2. The service publishes a domain event, often through a transactional outbox.
3. Consumers update read models, caches, or search indexes.
4. Queries use those projections where their consistency characteristics are acceptable.

This allows read models to be denormalized without making them the source of truth. It also means projections can lag and require replay, reconciliation, or rebuild procedures.

---

## 4. Buyer and Seller Workflows

### 4.1 Buyer journey

```text
Browse / search
      |
      v
Product detail page: catalog + offers + availability
      |
      v
Add to cart
      |
      v
Checkout: address + shipping + payment authorization
      |
      v
Order accepted
      |
      v
Fulfillment and shipment tracking
      |
      v
Delivery, returns, and customer support
```

The product page may combine data from catalog, seller offers, price, availability, reviews, and recommendations. Aggregation should have timeouts and fallback behavior so that a slow noncritical dependency does not block the entire page.

Checkout should communicate an accurate state to the buyer. An accepted order, authorized payment, reserved inventory, and shipped package are distinct state transitions; a success response should not imply that downstream work is already complete.

### 4.2 Seller journey

```text
Seller onboarding and verification
      |
      v
Create or match product catalog entries
      |
      v
Publish offer, price, and fulfillment method
      |
      v
Update inventory or send stock to fulfillment
      |
      v
Receive order and fulfill (seller or platform)
      |
      v
Settlement, performance reporting, and support
```

Seller APIs and portals require authentication, authorization, input validation, rate limits, audit trails, and idempotent update semantics. Bulk catalog and inventory uploads are commonly handled asynchronously with validation reports rather than through long-running synchronous requests.

---

## 5. Catalog, Offers, and Search

### 5.1 Product catalog

Products can have heterogeneous attributes: a laptop has processor and memory fields, while apparel has size and color. The catalog model therefore needs a stable shared identity and category-specific attributes.

Common design options include:

- A relational write model for strong constraints and controlled product changes.
- A document or key-value read model for denormalized product-page retrieval.
- Parent-child relationships for variations such as size or color.
- Media storage and CDN delivery for images and other assets.
- Validation and moderation workflows for seller-submitted content.

The specific database choice depends on write consistency, query patterns, schema evolution, and operational capability.

### 5.2 Offers and seller inventory

A product and an offer are not the same entity. One product can have multiple seller offers, each with its own price, shipping terms, fulfillment method, and availability.

An offer model may include:

| Field | Example meaning |
|---|---|
| Product ID | Catalog item being offered |
| Seller ID | Seller providing the offer |
| Price and currency | Current seller price |
| Fulfillment method | Platform-fulfilled or seller-fulfilled |
| Shipping promise | Estimated delivery options |
| Inventory reference | Available quantity or inventory location |
| Offer status | Active, suppressed, or unavailable |

### 5.3 Search indexing

Search systems typically maintain a derived index optimized for text search, filters, facets, and ranking. Catalog changes are propagated to the index asynchronously. Search results may therefore be eventually consistent with the system of record.

Representative tools include OpenSearch or Elasticsearch. Re-indexing, schema migrations, relevance evaluation, and stale-document removal are important operational concerns.

---

## 6. Pricing and Buy Box Evaluation

A marketplace may select a featured offer using multiple factors, potentially including total price, delivery promise, availability, seller eligibility, and service performance. The exact policy and weights are business-specific and may change.

```text
Seller offer updates / availability changes
                   |
                   v
              Event stream
                   |
                   v
         Stateful rules / evaluation
                   |
          +--------+--------+
          |                 |
          v                 v
  Read-optimized cache   Persistent offer state
          |
          v
 Product-page response
```

### 6.1 Event processing

Seller APIs, internal systems, or approved competitive data sources can produce offer and pricing updates. A stream processor such as Apache Flink or a managed streaming service can maintain keyed state and recompute derived results as inputs change.

The resulting featured offer or price projection can be written to a low-latency read store. This projection should be treated as derived data; the authoritative offer and pricing records remain owned by their domain services.

### 6.2 Design considerations

- Partition events by a key that preserves required ordering, such as product or offer ID.
- Handle duplicate and out-of-order events.
- Define event-time and late-event behavior.
- Keep pricing policy versioned and auditable.
- Invalidate or refresh caches when relevant inputs change.
- Avoid implying that a cache write-back is the durable source of truth.

---

## 7. Cart, Checkout, and Inventory Reservation

### 7.1 Shopping cart

A cart is a high-churn buyer-specific collection of selected items. It can be represented as a document or key-value record keyed by buyer ID, with a cache for active sessions where useful.

The persistent cart remains authoritative; cache entries should be recoverable and have clear expiration and invalidation behavior. Cross-region replication behavior and conflict resolution must be chosen based on user expectations and the underlying store's consistency model.

### 7.2 Checkout orchestration

An order may involve address validation, shipping selection, inventory reservation, payment authorization, fraud checks, and order persistence. These operations should not be modeled as one distributed database transaction spanning every service.

```text
Checkout request (idempotency key)
       |
       v
Validate cart, address, and eligibility
       |
       v
Reserve inventory -----> Reservation accepted or rejected
       |
       v
Authorize payment ------> Authorization accepted or declined
       |
       v
Persist order and publish OrderPlaced event
       |
       v
Fulfillment, confirmation, and downstream processing
```

The exact sequence depends on business rules. Each step needs explicit state, timeout handling, retries, and compensation for failures.

### 7.3 Inventory concurrency

When stock is scarce, many buyers may attempt to purchase the final units concurrently. A conditional atomic update can ensure that stock is reserved only when sufficient quantity remains.

```sql
UPDATE inventory
SET available_quantity = available_quantity - :requested_quantity,
    version = version + 1
WHERE product_id = :product_id
  AND available_quantity >= :requested_quantity;
```

The syntax is illustrative; actual conditional-write support varies by database. If the update affects no row, the reservation did not succeed and the application must return or recover from an appropriate state.

### 7.4 Reservation lifecycle and saga

1. Create a reservation with an expiry time.
2. Record the reservation state durably.
3. Continue checkout and payment authorization.
4. Confirm the reservation when the order reaches the appropriate state.
5. Release the reservation if checkout expires or a compensating workflow is required.

A saga coordinates these multi-service steps using forward actions and compensating actions. A reservation timeout should be based on business and operational requirements, not a hard-coded universal duration.

### 7.5 Idempotency and transactional outbox

- **Idempotency key:** A client-supplied key lets the service recognize retries of the same logical operation and return the prior result rather than creating duplicate orders or charges. The key must be scoped, persisted, and retained for an appropriate period.
- **Transactional outbox:** Write the domain update and an event record in the same local transaction. A publisher then delivers the event to the broker. Consumers should also be idempotent because message delivery can be repeated.

These patterns reduce duplicate effects; they do not create a global exactly-once transaction across every service.

---

## 8. Supply Chain, Fulfillment, and Delivery

E-commerce connects digital orders to physical inventory and transportation. The following is a conceptual flow; a real logistics network has additional constraints and specialized systems.

```text
Seller inbound shipment
       |
       v
Receiving and inventory placement
       |
       v
Order allocation to fulfillment location
       |
       v
Picking -> packing -> sortation
       |
       v
Carrier / delivery partner handoff
       |
       v
Tracking, delivery confirmation, returns
```

### 8.1 Inbound planning and placement

Demand forecasts and inventory position can help decide where stock should be sent. Batch analytics or distributed processing frameworks such as Spark may support planning workloads. The allocation decision balances expected demand, transportation cost, capacity, and service promises.

### 8.2 Warehouse management and robotics

Warehouse management systems track item locations, work queues, labor, and equipment. Automated guided vehicles or other robotics may be coordinated by specialized control systems with safety constraints and local fail-safe behavior.

The original notes mention actor-style concurrency frameworks, C++ or Rust control software, MQTT, and UDP as examples. These are possible implementation choices, not a verified description of any particular company's warehouse stack. Safety-critical controls require deterministic failure handling and must not rely solely on a best-effort network.

### 8.3 Order allocation and fulfillment routing

An allocation service can select a fulfillment node based on:

- Item availability and reservation state.
- Warehouse capacity and operating status.
- Distance, transport cost, and delivery promise.
- Carrier schedules and pickup cutoffs.
- Split-shipment cost and customer preferences.

The network may be modeled as a graph, but the optimization problem includes changing constraints and is not necessarily a simple shortest-path calculation. Once a node is selected, the system creates work for picking, packing, and shipment services.

### 8.4 Last-mile routing and telemetry

Delivery planning is a vehicle-routing problem with constraints such as package volume, route duration, time windows, driver availability, and traffic. Heuristics or optimization services may produce routes, while telemetry and tracking events update estimated delivery times.

Mobile applications can send location or status updates through authenticated APIs or event streams. Collection should be limited to what is needed and governed by privacy and retention policies.

---

## 9. Personalization and Recommendations

Recommendation systems commonly separate candidate generation from ranking. The technologies below are representative examples rather than confirmed details of a specific company's stack.

```text
User interactions
       |
       +--> Event stream --> real-time feature updates
       |
       +--> Data lake --> offline model training
                             |
                             v
                        Model registry
                             |
                             v
Online features --> Candidate retrieval --> Ranking --> Product page
```

### 9.1 Event and feature pipeline

Clicks, searches, cart additions, and purchases can be emitted as events. A stream processor updates recent behavioral features, while historical events are stored for analytics and training. A feature store can expose online and offline feature views, provided that feature definitions remain consistent between training and serving.

Representative components may include Kafka or Kinesis for ingestion, Flink for stream processing, object storage such as S3 with a table format such as Iceberg for history, and a feature-store implementation such as Feast or a managed equivalent.

### 9.2 Candidate generation and ranking

1. **Candidate generation:** A two-tower model can encode a user and items into a shared vector space. Approximate nearest-neighbor search can retrieve a manageable candidate set.
2. **Ranking:** A more expressive model scores candidates using context such as price, inventory, user preferences, and session activity.
3. **Business rules:** Eligibility, availability, safety, diversity, and policy constraints may filter or adjust results.

Potential vector search technologies include Redis Vector Search, Milvus, or other ANN indexes. Potential training or serving platforms include SageMaker or distributed compute frameworks. A knowledge graph such as Amazon Neptune can support relationship-oriented queries, but it is not required for every recommendation system.

Latency goals such as retrieving hundreds of candidates in a few milliseconds are workload targets to benchmark, not guarantees derived from a particular database or model.

---

## 10. Caching and Content Delivery

Caching reduces repeated work and protects backends, but every cache needs a defined freshness and failure strategy.

| Layer | Typical use | Important considerations |
|---|---|---|
| CDN / edge cache | Static assets and carefully selected public content | Cache keys, invalidation, privacy, geographic behavior |
| Application cache | Frequently read product or offer projections | TTL, stampede prevention, stale-read tolerance |
| Distributed in-memory cache | Hot records, sessions, or derived results | Eviction, replication, failover, memory limits |
| Database buffer cache | Frequently accessed persisted data | Managed by database and workload behavior |

Redis, Memcached, and managed services such as ElastiCache are common options. Cache misses must not create uncontrolled surges against the system of record; request coalescing, rate limits, and backpressure can help.

---

## 11. Data Stores and Distributed Data Patterns

No single data store is optimal for every workload. Polyglot persistence chooses stores according to access patterns, consistency requirements, and operational skills.

| Data need | Possible store type | Example technologies |
|---|---|---|
| Financial or relational transactions | Relational database | PostgreSQL, MySQL, Aurora |
| High-volume key-based records | Key-value or document database | DynamoDB, Cassandra |
| Full-text search and facets | Search index | OpenSearch, Elasticsearch |
| Cache and ephemeral state | In-memory key-value store | Redis, Memcached |
| Relationships and graph traversal | Graph database | Neptune |
| Event history and analytics | Object storage / lakehouse | S3 with Iceberg or similar |

### 11.1 Partitioning and replication

- **Partitioning / sharding:** Distributes records across nodes by a partition key. Poor key choice can create hot partitions or uneven storage.
- **Replication:** Maintains copies for availability and read scaling. Replicas may lag behind writes.
- **Read/write separation:** Can reduce primary read load but requires handling replica lag and read-after-write expectations.
- **Global replication:** Helps serve users in multiple regions but requires explicit conflict and consistency semantics.

Avoid simplistic assumptions such as splitting user IDs alphabetically. Real shard keys are selected to distribute load and support the dominant access patterns.

### 11.2 Consistency choices

Use the consistency level required by the business operation. Search indexes and recommendation features can often be eventually consistent. Inventory reservation, order transitions, and payment state require carefully defined atomic operations and reconciliation.

Global tables and distributed databases have product-specific consistency semantics; cross-region replication should not be assumed to provide immediate, conflict-free global writes.

---

## 12. Resilience, Security, and Observability

### 12.1 Resilience patterns

- **Timeouts:** Bound how long a request waits for a dependency.
- **Retries:** Retry only safe operations, with backoff and jitter; avoid retry storms.
- **Circuit breakers:** Temporarily stop calls to a failing dependency.
- **Bulkheads:** Isolate thread pools, connection pools, or concurrency limits by dependency.
- **Backpressure and load shedding:** Keep the system within capacity instead of accepting work it cannot process.
- **Graceful degradation:** Serve a basic product page or non-personalized recommendations if an optional personalization dependency fails.
- **Cell-based isolation:** Partition infrastructure into independent cells or failure domains to reduce blast radius.
- **Reconciliation:** Compare related state across services and repair discrepancies.

Libraries such as Resilience4j and platform-level controls can implement some of these patterns. Fallbacks should be safe, explicit, and observable.

### 12.2 Security controls

- Authenticate buyers, sellers, and service-to-service callers.
- Enforce least-privilege authorization at the service and data layers.
- Use TLS in transit and encryption at rest where appropriate.
- Protect payment data and use a payment provider or tokenization strategy suited to compliance requirements.
- Validate and sanitize inputs; apply bot and abuse controls.
- Record security-relevant events without logging secrets or unnecessary personal data.

### 12.3 Observability

Requests may cross many synchronous and asynchronous boundaries. Use a trace or correlation identifier and propagate it through HTTP, gRPC, and message metadata. OpenTelemetry and AWS X-Ray are examples of tracing tools.

Operational signals should include:

- **Metrics:** Request rate, error rate, latency percentiles, queue lag, saturation, cache hit rate.
- **Logs:** Structured, redacted records with useful correlation fields.
- **Traces:** Service spans and asynchronous producer-consumer relationships.
- **Business indicators:** Checkout conversion, payment declines, reservation failures, shipment delays.

Alerts should focus on user impact and actionable symptoms, not merely individual host conditions.

---

## 13. Redis Vector Search and Java WebFlux Deep Dive

This section retains the original technical exploration as an illustrative Redis Vector Search design. It is not a statement that a specific marketplace uses this topology.

### 13.1 Example topology

```text
API gateway / service mesh
          |
          v
Java WebFlux service (Netty)
          |
          +---- Redis Cluster shard 1 (index + vectors)
          +---- Redis Cluster shard 2 (index + vectors)
          +---- Redis Cluster shard 3 (index + vectors)
                       |
                       +---- replicas, depending on deployment
```

Redis Stack or a managed Redis-compatible service may support vector indexes such as HNSW. Index layout, memory usage, threading, replication, and query routing depend on the exact Redis version and product.

### 13.2 Memory sizing example

For a 1,536-dimensional vector stored as 32-bit floating-point values:

```text
1,536 dimensions x 4 bytes = 6,144 bytes per raw vector
10,000,000 vectors x 6,144 bytes = 61.44 GB of raw vector values
```

This excludes keys, metadata, allocator overhead, replicas, index graph links, and operational headroom. HNSW graph parameters and the data distribution affect index size and search quality, so total memory must be measured using representative data and the target deployment.

### 13.3 Query fan-out and cluster routing

When an index spans multiple shards, a query may require shard fan-out or routing to the relevant partitions. The client or proxy merges partial results. The cost depends on index distribution, query routing, filtering, network latency, and the number of shards involved.

Redis Cluster commonly uses 16,384 hash slots for key placement. This does not by itself specify how a vector index is distributed or guarantee a particular query latency.

### 13.4 Replication and read routing

Replication can improve availability and may help with read scaling, subject to the product's consistency and routing behavior. A just-written vector may not be immediately visible on a replica. Choose whether search queries can tolerate that lag, and test failover and resynchronization behavior.

### 13.5 Java WebFlux request path

Spring WebFlux commonly runs on Netty and uses non-blocking I/O. A conceptual request path is:

```text
Network request
      |
      v
Netty event loop
      |
      +--> non-blocking application pipeline
      |
      +--> reactive Redis client (for example, Lettuce)
                    |
                    v
             Redis cluster nodes
      |
      v
Reactive response serialization
```

Operational guidelines:

1. Keep event-loop work short. Do not run blocking calls or long CPU tasks on event-loop threads.
2. Use a bounded scheduler for genuinely CPU-heavy transformations, measuring its queue and saturation.
3. Reuse pooled client connections and configure timeouts and concurrency limits.
4. Refresh cluster topology using supported client mechanisms and test node movement.
5. Apply backpressure and bound in-flight requests so a slow cluster cannot exhaust application memory.

Non-blocking I/O improves connection concurrency for I/O-bound services; it does not make CPU-bound work faster or remove downstream capacity limits.

### 13.6 Operational tuning checklist

| Area | Example to validate | Operational concern |
|---|---|---|
| Linux socket backlog | `net.core.somaxconn` | Align OS, container, and service listen backlogs. |
| Memory and persistence | Redis memory policy and snapshot configuration | Leave headroom for persistence and replication; test fork behavior. |
| Vector metric | Inner product or cosine distance | Ensure the metric matches training and vector normalization. |
| Redis I/O threads | Version- and workload-specific setting | Benchmark protocol I/O benefit and CPU contention. |
| Lettuce topology refresh | Supported adaptive refresh settings | Avoid stale routing and excessive refresh traffic. |
| TCP behavior | `TCP_NODELAY` where appropriate | Benchmark latency and packet overhead rather than assuming improvement. |

Kernel parameters and Redis settings are deployment-specific. Validate changes against the target OS, managed-service restrictions, and measured workload.

---

## 14. Serving Millions of Concurrent Users

“Concurrent users” is not the same as requests per second. Capacity planning should estimate request arrival rate, request duration, connection behavior, payload size, cache hit ratio, and downstream work.

### 14.1 Traffic distribution and backends

- DNS and CDNs absorb or cache suitable traffic near users.
- Layer 4 and Layer 7 load balancers distribute connections and requests.
- Horizontal application scaling adds instances, while autoscaling reacts to measured demand with nonzero delay.
- Distributed caches protect databases from repeated reads.
- Queues isolate bursty work and allow background workers to process at a controlled rate.
- Database partitioning, indexes, replicas, and query optimization support data access at scale.

Representative technologies include CloudFront, Cloudflare, Akamai, NGINX, HAProxy, AWS ALB/NLB, Redis, Memcached, Kafka, RabbitMQ, and SQS.

### 14.2 Java REST API concurrency

#### Thread-per-request model

Traditional servlet stacks, such as Spring MVC on Tomcat, commonly use a worker thread per active request. This is straightforward, but blocked threads waiting on I/O consume resources. The right thread-pool size depends on workload and deployment; a fixed range is not universally appropriate.

#### Non-blocking reactive model

Spring WebFlux with Netty can use a small event-loop group to multiplex many I/O-bound connections. It is effective when the entire call chain is non-blocking. A blocking database driver or blocking call in the reactive path can still cause event-loop starvation.

#### Connection pools and JVM behavior

- Reuse database and HTTP connections through appropriately sized pools.
- Size pools with downstream capacity in mind; larger pools can increase contention.
- Monitor heap, allocation rate, garbage-collection pauses, and CPU saturation.
- Modern collectors such as ZGC or Shenandoah may fit latency-sensitive workloads, but must be benchmarked with the selected JDK and heap size.

#### Load protection

Rate limiting, circuit breaking, bulkheads, timeouts, and backpressure protect services from overload. When capacity is exhausted, reject or shed work predictably instead of allowing queues and latency to grow without bound.

### 14.3 Distributed architecture patterns

| Pattern | Benefit | Important trade-off |
|---|---|---|
| Microservices | Independent ownership and scaling by domain | Network failures, operational complexity, distributed data |
| Event-driven architecture | Decouples producers and consumers; buffers work | Eventual consistency, duplicate delivery, ordering |
| CQRS | Read models optimized independently from writes | Projection lag and rebuild complexity |
| Sharding and partitioning | Spreads data and throughput across nodes | Hot keys, cross-shard operations, resharding |
| Read replicas | More read capacity and regional access | Replication lag and stale reads |
| Service mesh | Standardizes service traffic and telemetry | Extra operational and runtime complexity |
| Cell-based architecture | Limits failure blast radius | Cell placement, migration, and cross-cell workflows |

---

## 15. Interview Discussion Checklist

For a principal or staff architect interview, explain decisions and trade-offs rather than presenting a technology list.

1. **Clarify requirements:** Traffic profile, peak-to-average ratio, latency SLOs, availability, geography, consistency, and cost.
2. **Define domain boundaries:** Identify ownership of catalog, offers, inventory, orders, payments, and fulfillment.
3. **Draw request and event flows:** Separate synchronous user-facing work from asynchronous processing.
4. **Choose data stores by access pattern:** Explain the system of record, read projections, partition keys, and indexing.
5. **Protect checkout correctness:** Discuss idempotency, inventory reservation, payment authorization, order state, and saga compensation.
6. **Plan for overload:** Describe backpressure, rate limits, timeouts, load shedding, and graceful degradation.
7. **Address failure recovery:** Cover retries, outbox, duplicate events, replay, reconciliation, and regional failover.
8. **Explain observability and security:** Include trace propagation, operational metrics, least privilege, privacy, and payment protection.
9. **State consistency boundaries:** Identify which views may be stale and which state transitions require atomicity.
10. **Quantify and validate:** Estimate capacity, benchmark bottlenecks, test failure modes, and revisit assumptions as measurements arrive.

---

## 16. Further Reading

These references provide general background on distributed architecture. The original notes included them as supporting material; the document does not endorse every claim made by third-party articles.

- [AWS: What is distributed architecture?](https://aws.amazon.com/what-is/distributed-architecture/)
- [vFunction: Distributed architecture overview](https://vfunction.com/blog/distributed-architecture/)
