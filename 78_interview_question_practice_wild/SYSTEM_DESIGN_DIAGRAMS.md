# Architecture diagram index

<span class="practice-page"></span>

[Chapter guide](README.md) · [Complete RAG and agent answers](08_rag_and_agents.md) · [Complete ML system-design answers](09_ml_system_design.md)

**317 design-oriented question entries** now have an architecture, component or workflow diagram in their answer. W408 keeps its two bespoke cart diagrams. W537 has separate order-placement and quote-serving paths. The four expanded guides contain 40 full explanations. These and related design sections include 38 foundation diagrams.

A diagram is a starting architecture to explain and defend, not a claim that each box must be a deployed microservice. A general interview round without a fixed product uses an answer-planning view; it does not invent a missing product specification. Coding and mathematical questions without a system-design task are outside this diagram index.

For each drawing: trace one request, identify the state owner, discuss a duplicate or timeout, and say how recovery preserves the invariant. Diagrams are local SVG files and need no online rendering service.

## API limits, quotas and overload

- [W231. Design a Distributed Rate Limiter with Redis Lua](questions/W231.md)
- [W232. Design a Network Throttling / Rate-Limiting System](questions/W232.md)
- [W233. Design a Quota System](questions/W233.md)
- [W234. Design an API Rate Limiter](questions/W234.md)
- [W235. Design and Implement a Rate Limiter](questions/W235.md)
- [W236. Rate Limiter Algorithm Design](questions/W236.md)
- [W237. Rate Limiter Design (OOD)](questions/W237.md)
- [W238. Rate Limiter for Expensive API with User-Configurable Monthly Quota](questions/W238.md)
- [W239. Rate Limiter — Inbound + Outbound](questions/W239.md)

## Algorithms, simulations and geometry

- [W243. Versioned Social Network with Snapshots](questions/W243.md)
- [W368. Python Data Structure Design for Game Metadata and Related Discord Servers](questions/W368.md)
- [W369. Simulation-style coding problem with scaling follow-up (mini system design)](questions/W369.md)

## Business rules and transactional services

- [W370. Design a Food Review System with Reward Bonus](questions/W370.md)
- [W496. Design a Sales Tax Engine (Address + Date to Tax)](questions/W496.md)

## Coding, concurrency and application structure

- [W247. Thread-Safe Cache with Lock and Condition Variable](questions/W247.md)
- [W248. Thread-Safe Job Scheduler with Cancellation](questions/W248.md)
- [W372. Design a Polling Schedule with a Boot State and a Live State](questions/W372.md)

## Configuration, identity and authorization

- [W249. Design a File-Sharing Permission Control System](questions/W249.md)
- [W250. Design an ACL Authorization Service](questions/W250.md)
- [W251. Design an IP Blocking Policy Decision Service](questions/W251.md)
- [W373. Cross-Domain SSO After an Acquisition, and OIDC Across Sites](questions/W373.md)
- [W374. Deep Dive — What Happens When You Access a URL](questions/W374.md)
- [W375. Design a Feature Flag System](questions/W375.md)
- [W376. Design a Settings Store with User and Team Sync](questions/W376.md)
- [W377. Designing HTTPS Load Balancer with SSL, HA, and Scaling](questions/W377.md)
- [W378. Global Config Deployment System](questions/W378.md)
- [W379. Understanding Cross-Origin Issues and Their Solutions](questions/W379.md)
- [W380. Website Access: DNS, CDN, Protocols, Service Discovery, Authentication](questions/W380.md)

## Crawling and fan-out pipelines

- [W381. Book Seller External Fan-Out Broker Service](questions/W381.md)
- [W382. Design a Distributed Web Crawler](questions/W382.md)
- [W383. Design a Distributed Web Crawler](questions/W383.md)
- [W384. Design a Web URL Crawler at Scale](questions/W384.md)
- [W385. Image Link Crawler Service](questions/W385.md)
- [W386. Priority-Based Rate-Limited Web Crawler Design](questions/W386.md)
- [W387. Wikipedia Internet Archive / Web Crawler](questions/W387.md)

## Distributed consistency and recovery

- [W254. Implement a Leader-Follower Distributed System with Membership, Failover Detection, and Leader Election (Practical pair-programming)](questions/W254.md)
- [W388. Design a Read-Once Password Sharing Service](questions/W388.md)

## Evaluation, calibration and data quality

- [W039. Build an LLM Evaluation Harness with Ollama Integration](questions/W039.md)
- [W054. Evaluating an LLM Agent with Subjective, Nondeterministic Outputs](questions/W054.md)
- [W100. MLE Deployment, Monitoring & Latency Optimisation](questions/W100.md)
- [W139. Design an A/B Testing / Experimentation Platform](questions/W139.md)
- [W257. Video-Keyword Association Review System](questions/W257.md)

## Feeds, notifications and social graphs

- [W258. Design a Large-Scale News Feed](questions/W258.md)
- [W259. Design a Status/Feed App](questions/W259.md)
- [W260. Design Instagram](questions/W260.md)
- [W261. Design Instagram / Facebook News Feed](questions/W261.md)
- [W262. Design Twitter: Posting, Retrieval and Search](questions/W262.md)
- [W263. Designing Nextdoor Newsfeed User Interaction and Real-Time Feed System](questions/W263.md)
- [W264. Designing YouTube](questions/W264.md)
- [W265. Product Feed and Shopping Recommendation System](questions/W265.md)
- [W266. Snap Story or Spotlight Feed Backend](questions/W266.md)
- [W389. Design a Notification / Alert Fan-Out System](questions/W389.md)
- [W390. Design a Notification System with Preferences and Fan-out](questions/W390.md)
- [W391. Notification System](questions/W391.md)
- [W392. Notification System (Event + Batch)](questions/W392.md)
- [W393. Notification System — Push to Followers on New Post](questions/W393.md)
- [W394. Stock Price Alert Notification System](questions/W394.md)

## Frontend and full-stack systems

- [W509. Design a Website for 100,000 Concurrent Users with Live Comments](questions/W509.md)
- [W511. Frontend System Design: Autocomplete Search Input](questions/W511.md)
- [W512. Frontend System Design: Infinite-Scroll Image Feed](questions/W512.md)
- [W513. Frontend System Design: Signup Form with A/B Testing](questions/W513.md)

## Games, simulations and object design

- [W395. Daily Puzzle Submission and Leaderboard System](questions/W395.md)
- [W396. Game Leaderboard System Design](questions/W396.md)
- [W514. Design a Matchmaking Service for Multiplayer Games](questions/W514.md)
- [W515. Design a Real-Time Multiplayer Game Backend](questions/W515.md)
- [W516. Design an Online Chess Platform (Matchmaking, Live Games, Clocks)](questions/W516.md)
- [W517. Minesweeper Game Design (OOD)](questions/W517.md)
- [W518. System Design: Chess Game](questions/W518.md)

## Geospatial, dispatch and tracking systems

- [W267. Distributed Device Control](questions/W267.md)
- [W397. Design a Pickup Area Driver Queue](questions/W397.md)
- [W398. Design a Waymo-like Autonomous-Vehicle Ride-Hailing App](questions/W398.md)
- [W399. Design Multi-Carrier Package Delivery Routing System](questions/W399.md)
- [W400. ML System Design: Choose Passenger Drop-off Location](questions/W400.md)
- [W401. Perishable-Goods Tracking & Location System](questions/W401.md)
- [W402. Query System — Time + Geo Filtered User Activity](questions/W402.md)
- [W403. Ride-Hailing System Design (Uber)](questions/W403.md)
- [W404. Rider-Driver Matching System](questions/W404.md)

## Inventory, reservations and marketplaces

- [W405. Airbnb Booking System Design](questions/W405.md)
- [W406. Airline Booking System: Search, Seat Hold, Purchase and Cancel](questions/W406.md)
- [W407. Design a Hotel Booking System with External Inventory](questions/W407.md)
- [W408. Design a Shopping Cart Service With Inventory Consistency](questions/W408.md)
- [W409. Design a Ticketing System with Flash-Sale Concurrency](questions/W409.md)
- [W410. Design an Event Ticket Marketplace: Seat Holds and On-Sale Traffic](questions/W410.md)
- [W411. Design an Expedia-Style Hotel Reservation System](questions/W411.md)
- [W412. Design an Online Auction System with Concurrent Bidding](questions/W412.md)
- [W413. Design Ticket Booking (Ticketmaster)](questions/W413.md)
- [W414. Design Ticketmaster - Event Booking and Seat Reservation](questions/W414.md)
- [W415. Million-User Flash Sale System](questions/W415.md)
- [W416. Online Shopping Cart System](questions/W416.md)
- [W417. Realtime Auction Bidding System](questions/W417.md)
- [W418. Shopping Cart Latency and Inventory](questions/W418.md)
- [W419. Ticketmaster-Style Seat Booking Design](questions/W419.md)
- [W420. Waitlist System Design](questions/W420.md)

## Jobs, concurrency and durable workflows

- [W268. Async Job Scheduler](questions/W268.md)
- [W269. Design a Distributed Job Scheduler with Cron, Pause, Resume](questions/W269.md)
- [W270. Design a Distributed Task Scheduler](questions/W270.md)
- [W271. Design a Multi-Resource Loader/Scheduler](questions/W271.md)
- [W272. Design an Async Job / Task System](questions/W272.md)
- [W273. Distributed Job Scheduler / Calendar Service](questions/W273.md)
- [W274. Job Scheduler System Design](questions/W274.md)
- [W275. Job Scheduler with Task Dependencies (DAG Executor)](questions/W275.md)

## LLM serving and scheduling

- [W026. Design a GPU Inference Serving System](questions/W026.md)
- [W030. Design ChatGPT: A Conversational LLM Serving Platform](questions/W030.md)
- [W031. ML System Design: Inference Serving with Back-of-Envelope Capacity Planning](questions/W031.md)
- [W111. Design a ChatGPT-like Conversational AI System](questions/W111.md)
- [W144. Design an API for Embedding and Classification Results](questions/W144.md)
- [W145. Design Sora Video Generation Scheduling with Preemption](questions/W145.md)

## Logs, metrics and observability

- [W276. Design a Distributed Metrics System](questions/W276.md)
- [W277. Design a Metric Counter Library](questions/W277.md)
- [W278. Design a Real-Time Log Processing System](questions/W278.md)
- [W279. Design a Real-Time Metrics Monitoring System](questions/W279.md)
- [W280. Design a User Behavior / Metrics Monitoring Aggregator](questions/W280.md)
- [W281. Design an Observability Logging System (Emit and Query)](questions/W281.md)
- [W282. Metrics and Monitoring Platform](questions/W282.md)
- [W283. Metrics Collection, Alerting & Click Counters](questions/W283.md)
- [W286. Telemetry Collector and GPU Utilization Dashboard](questions/W286.md)
- [W421. Core Dump Collection System Design](questions/W421.md)
- [W422. Cross-Platform Client Logging Library](questions/W422.md)
- [W423. Design an Audit Logs Service](questions/W423.md)
- [W424. Environment-specific Logging Without Code Modifications](questions/W424.md)
- [W425. Event Logger with batching, fsync, and group commit follow-ups](questions/W425.md)

## ML coding and gradients

- [W288. Parallel Batch Image Transformation Pipeline](questions/W288.md)

## ML infrastructure and distributed training

- [W029. Design a GPU Scheduling Platform](questions/W029.md)
- [W033. Design a Distributed Training Platform for Foundation Models](questions/W033.md)
- [W034. Distributed LLM Training Data Pipeline](questions/W034.md)
- [W037. Design Model Weight Distribution](questions/W037.md)
- [W038. Design a Distributed AI Model Downloader](questions/W038.md)
- [W150. CI/CD Job Scheduler Design using Kubernetes and Docker](questions/W150.md)
- [W151. Cloud IDE](questions/W151.md)
- [W152. Compute Resource Scheduling for ML Jobs](questions/W152.md)
- [W153. Design a Hosted ML Notebook Platform](questions/W153.md)
- [W154. Design a Job Scheduler / ETL Pipeline](questions/W154.md)
- [W155. Design a Lab-Wide Experiment Platform for Embodied AI](questions/W155.md)
- [W156. Design a Multistage CPU and GPU Job Scheduler](questions/W156.md)
- [W157. GPU Cloud VM Manager - Provider-Agnostic CLI and Fleet Manager](questions/W157.md)
- [W158. HPC Host Control Plane and Job Coordination](questions/W158.md)
- [W159. ML Platform Portal Full-Stack Design](questions/W159.md)
- [W289. Design a Release Deployment Workflow with Staged Rollout](questions/W289.md)
- [W290. Kubernetes Controller for Pod Balance](questions/W290.md)

## Messaging and collaborative systems

- [W293. Design a ROS-like Pub/Sub System](questions/W293.md)
- [W294. Design an In-Process Pub-Sub Messaging Broker (OOD)](questions/W294.md)
- [W427. Build a Discord-like TCP Chat Server with Slash Commands, Rooms, Reconnection, and Persistence](questions/W427.md)
- [W428. Design a 1:1 Chat System (Pub/Sub, Partitioning, Multi-Region)](questions/W428.md)
- [W429. Design a Chat Application with Group and Offline Support](questions/W429.md)
- [W430. Design a Collaborative Spreadsheet (Google Sheets)](questions/W430.md)
- [W431. Design a Real-Time Instant Messaging Application with Media Support](questions/W431.md)
- [W432. Design a Slack-like Chat System with Deletion Semantics](questions/W432.md)
- [W433. Group Chat / Messaging System Design](questions/W433.md)
- [W434. Real-Time Collaborative Code Editor](questions/W434.md)
- [W435. Real-Time Comment Threads (FigJam / Figma)](questions/W435.md)
- [W436. Subreddit Live Chat System](questions/W436.md)

## Payments, ledgers and billing

- [W437. Credit Card Service with Points Earning and Redemption System](questions/W437.md)
- [W438. Credit Card System Design (OOD)](questions/W438.md)
- [W439. Design a Delayed Virtual-Currency Payment Scheduler](questions/W439.md)
- [W440. Design a Highly Reliable Account Balance System](questions/W440.md)
- [W441. Design a Ledger Service with Merchant Balance](questions/W441.md)
- [W442. Design a Loan Payment Distribution System](questions/W442.md)
- [W443. Design a Multi-Chain Crypto Ledger & Wallet History](questions/W443.md)
- [W444. Design a Payment Gateway System](questions/W444.md)
- [W445. Design a Payment Webhook Delivery System](questions/W445.md)
- [W446. Design a Stripe-like Payment Platform](questions/W446.md)
- [W447. Design a Wallet with Ledger and Idempotent Transfers](questions/W447.md)
- [W448. Design Capital One's Online Banking Application](questions/W448.md)
- [W449. Subscription Billing System for 300M Subscribers](questions/W449.md)
- [W450. Venmo-Style P2P Payment Data Model](questions/W450.md)
- [W519. Charity Donation System](questions/W519.md)
- [W520. Design a 3-Day Donation Service](questions/W520.md)

## Production ML and product design

- [W059. ML System to Detect Harmful LLM Content](questions/W059.md)
- [W060. Designing a Harmful Content Detection System](questions/W060.md)
- [W061. Design an ML Fraud Detection System](questions/W061.md)
- [W062. Design an Account Takeover Prediction System](questions/W062.md)
- [W063. ML System Design: Search & Ranking](questions/W063.md)
- [W065. Video Recommendation](questions/W065.md)
- [W066. Homepage Video Recommendation System](questions/W066.md)
- [W067. Design Meta Ads Ranking](questions/W067.md)
- [W068. Design App Store Search and Ranking](questions/W068.md)
- [W069. Airbnb Experiences — Search Ranking (ML Design)](questions/W069.md)
- [W070. ML System Design: Restaurant / Store Recommendation](questions/W070.md)
- [W162. Content Moderation System Design](questions/W162.md)
- [W163. Design a Game Recommendation System](questions/W163.md)
- [W164. Design a Nearby-Place Recommender (Location-Aware)](questions/W164.md)
- [W165. Design a Recommendation System for a New Product (the Cold-Start Problem)](questions/W165.md)
- [W166. Design a Recommendation System for a New Product (the Cold-Start Problem)](questions/W166.md)
- [W167. Design a Reels Short-Video Recommender](questions/W167.md)
- [W168. Design a Restaurant Recommendation and Ranking System for a Food Delivery App](questions/W168.md)
- [W169. Design an Online Fraud Detection Pipeline](questions/W169.md)
- [W170. Design Harmful-Content Detection (Weapon Sales) at Scale](questions/W170.md)
- [W171. Design Multi-Source Notification Ranking](questions/W171.md)
- [W172. Design the Uber Eats Search System](questions/W172.md)
- [W173. Home Page — Search + Availability + Ranking](questions/W173.md)
- [W174. Improve Booking via Cover Photo Selection (ML Design)](questions/W174.md)
- [W175. Learning / Job Recommendation Ranking](questions/W175.md)
- [W176. Listing Lifetime Value — Estimation (ML Design)](questions/W176.md)
- [W177. ML News Recommendation System Design](questions/W177.md)
- [W178. ML System Design - Predict Item Category](questions/W178.md)
- [W179. ML System Design — Asset / Template Recommendation & Feed](questions/W179.md)
- [W180. ML System Design: Bad / Unsafe Content Detection](questions/W180.md)
- [W181. ML System Design: Behavior Prediction from Sensor + Camera Data](questions/W181.md)
- [W182. ML System Design: Notification Ranking & Ads CTR](questions/W182.md)
- [W183. ML System Design: Product Categorization / Taxonomy](questions/W183.md)
- [W184. ML System Design: Recsys, Chatbot, Image Classifier](questions/W184.md)
- [W185. ML System Design: Search, Ranking, Experimentation](questions/W185.md)
- [W186. Model Delivery-Time Prediction for Online Orders](questions/W186.md)
- [W187. Post-Comment Ranking System](questions/W187.md)
- [W188. Premium Product Recommendation System](questions/W188.md)
- [W189. Property Rental Search System](questions/W189.md)
- [W190. Real-Time Fraud Detection System](questions/W190.md)
- [W191. Ride Request Demand Prediction by Geohash and Premium Multiplier](questions/W191.md)
- [W192. Short Video Recommendation & Ranking](questions/W192.md)
- [W193. Snap Ads Ranking](questions/W193.md)
- [W194. Spam Email Detection: Signals, Model, and Metrics](questions/W194.md)
- [W195. Support Ticket Routing & Claiming System](questions/W195.md)
- [W304. Ads Audience Targeting / Custom Audience System](questions/W304.md)
- [W306. Comment-Prediction ML System](questions/W306.md)
- [W307. Design a Credit-Approval Risk Engine](questions/W307.md)
- [W308. Design a Scalable Ad Marketplace System](questions/W308.md)
- [W309. Design Image Copyright-Violation Detection](questions/W309.md)
- [W310. Design Robot Human-Avoidance (Open-Ended)](questions/W310.md)
- [W311. ML System Design: Game Genre Classification From Scratch](questions/W311.md)

## RAG, agents and application security

- [W042. RAG Notebook (ML Coding)](questions/W042.md)
- [W043. RAG Search Augmentation and Internal Chatbot](questions/W043.md)
- [W044. RAG Q&A Chatbot — ML / AI Technical Deep Dive](questions/W044.md)
- [W045. Design a Grounded AI Chatbot App](questions/W045.md)
- [W046. Design Siri's Grounded Response Generation](questions/W046.md)
- [W047. Layered Defenses Against LLM Jailbreaks](questions/W047.md)
- [W048. LLM SQL Chatbot Access Control and Query Safety](questions/W048.md)
- [W049. Design a Chatbot Personalization / Memory System](questions/W049.md)
- [W050. Design Prompt Playground](questions/W050.md)
- [W051. Live Coding: Replace a Heuristic Terminal Game with an LLM-Driven Agent](questions/W051.md)
- [W052. Agents / Coding with LLMs](questions/W052.md)
- [W053. Design an LLM-Agent System for Automation (Ticket Triage / Code Review)](questions/W053.md)
- [W055. Design Durable Execution for Long-Running Agent Queries](questions/W055.md)
- [W056. ML Model Design — Prompt-to-Design Generation](questions/W056.md)
- [W057. ML Design: Recruiter Hiring Pipeline Assistant](questions/W057.md)
- [W058. Design an LLM-Backed Customer-Service Chat System with Failure Handling](questions/W058.md)
- [W113. Agent Tool-Use System Design (AML Volcano Engine)](questions/W113.md)
- [W114. Agentic Workflow for 1-Hour Movie Generation](questions/W114.md)
- [W115. Design a Campaign Outreach Workflow Engine](questions/W115.md)
- [W116. Design a Customer-Service Chatbot for Capital One](questions/W116.md)
- [W117. Jira AI Auto-Fix Pipeline](questions/W117.md)
- [W213. AI Personalized Recruiter Message Generation](questions/W213.md)
- [W214. Design a Customer Review Page with AI Summarization](questions/W214.md)
- [W215. Design an AI Prompt Playground](questions/W215.md)
- [W216. Design an LLM Flight Search and Booking Agent](questions/W216.md)
- [W217. Design LinkedIn's Skills Extraction & Taxonomy Pipeline](questions/W217.md)
- [W218. Tweet Insight Pipeline with Rate-Limited LLM Classification](questions/W218.md)
- [W219. Twitter Insight Platform with Rate Limiting and Dockerization](questions/W219.md)

## SQL, data modeling and pipelines

- [W118. ML Feature Store](questions/W118.md)
- [W325. DAU / MAU Internal Analytics System](questions/W325.md)
- [W326. Design a Financial Dashboard Backend](questions/W326.md)
- [W327. Design an Ad Click Aggregation Pipeline](questions/W327.md)
- [W328. Design an Analytics Metrics Dashboard for ChatGPT / LLM Service](questions/W328.md)
- [W329. Design Coinbase Explore Real-Time Price Dashboard](questions/W329.md)
- [W344. WAL Log Enrichment CDC Pipeline at 1M writes/sec](questions/W344.md)

## Search and retrieval systems

- [W064. ML System Design: Dynamic K in Retrieval Stage](questions/W064.md)
- [W345. Design a Post Search Engine (Mini Elasticsearch)](questions/W345.md)
- [W346. Design Apple News Search (No ML, V1)](questions/W346.md)
- [W347. Design Post Search with Keyword Exclusion and Frequency Ranking](questions/W347.md)
- [W348. Design Search Autocomplete / Typeahead](questions/W348.md)
- [W349. Type-Ahead / Autocomplete Suggestions](questions/W349.md)
- [W471. Design a podcast publishing and feed system](questions/W471.md)
- [W472. Design a Topic-Based News Aggregator](questions/W472.md)
- [W473. Designing a Scalable News Aggregator System with Cassandra](questions/W473.md)

## Storage, indexes and caching

- [W474. Design a Distributed Blob Cache With a Separate Metadata Index](questions/W474.md)
- [W475. Design a High-Throughput Distributed Cache](questions/W475.md)
- [W476. Design an S3-Style Blob Store with Deduplication](questions/W476.md)
- [W477. Design Google Drive / Data Room](questions/W477.md)
- [W478. Design Object Storage / File Sync (Dropbox / S3)](questions/W478.md)
- [W479. Distributed File System Metadata Layer with Strong Consistency](questions/W479.md)
- [W480. Distributed Key-Value Store Design](questions/W480.md)
- [W481. File Backup System with Durable Manifest](questions/W481.md)
- [W482. In-Memory KV Cache with Write-Ahead Log](questions/W482.md)
- [W483. Single-Machine Key-Value Store with Filesystem Spill](questions/W483.md)

## Streaming counters and top-K

- [W350. Ads Frequency Cap and Limiter System](questions/W350.md)
- [W351. Design a Like / Unlike Counter Service at Scale](questions/W351.md)
- [W352. Design a Top-K trending items service](questions/W352.md)
- [W353. Design a Trending-Hashtags Detection System](questions/W353.md)
- [W354. Design State-Wide Temperature Sensor Ingestion](questions/W354.md)
- [W355. Near-Real-Time Friend-Played-Game Counter](questions/W355.md)
- [W356. Real-time status counting system design](questions/W356.md)
- [W357. Recent Like Count and Top Posts in a Sliding Window](questions/W357.md)
- [W358. Social Media Sentiment Tracking System](questions/W358.md)
- [W359. Streaming Social-Media Mentions and Aggregation](questions/W359.md)
- [W360. System Design — Event Ingestion + Top-K Aggregation](questions/W360.md)
- [W361. Top-K Popular Items Over a Rolling Window](questions/W361.md)
- [W362. Trending System / Top-K Hot Files](questions/W362.md)

## Trading and financial systems

- [W531. Design a Crypto Brokerage Order System](questions/W531.md)
- [W532. Design Pre-Trade Risk Checking for an Electronic Trading System](questions/W532.md)
- [W533. Low-Latency Multi-Exchange Buy/Sell System Design](questions/W533.md)
- [W534. Low-Latency Trading System Design](questions/W534.md)
- [W535. Low-Latency Trading System Design](questions/W535.md)
- [W536. Real-time Market Data Distribution System](questions/W536.md)
- [W537. Stock Trading / Real-Time Quote System Design](questions/W537.md)
- [W538. Stock Trading Agent Service with Order Deadlines](questions/W538.md)
- [W539. Trade Processor with Subscribe / Unsubscribe](questions/W539.md)

## Web services and API design

- [W484. Design a Concurrent Image Upload System](questions/W484.md)
- [W485. Design a URL Shortener](questions/W485.md)
- [W486. Design a URL Shortener (TinyURL)](questions/W486.md)
- [W487. Design a URL Shortener (TinyURL)](questions/W487.md)
- [W488. Design a URL Shortener with Aliases, Expiry and Click Analytics](questions/W488.md)
- [W489. Design Scalable TinyURL Service with One-to-One URL Mapping](questions/W489.md)
- [W490. URL Shortener With Click Tracking](questions/W490.md)

## Web services and object design

- [W363. L6 Staff System Design Loop](questions/W363.md)
- [W364. System Design for Full Stack Applications](questions/W364.md)
- [W491. Ads Demand Intake Data Modeling](questions/W491.md)
- [W492. Design Walmart Plus Membership System](questions/W492.md)
- [W493. Promotion Posting System Data Model](questions/W493.md)
- [W494. Tagging System REST API](questions/W494.md)
- [W495. Template & Instance System with Update Propagation](questions/W495.md)
- [W540. Coffee Ordering System Design](questions/W540.md)
- [W541. Concurrent User Management System Architecture Design](questions/W541.md)
- [W542. Design a Pizza Ordering and Fulfillment System (OOD)](questions/W542.md)
- [W543. Design a Todo-List Full-Stack Service](questions/W543.md)
- [W544. Library Book Management System Design and API Architecture](questions/W544.md)
- [W545. Object-Oriented Design Round](questions/W545.md)
- [W546. Runnable Stack Overflow / Quora REST Service](questions/W546.md)
- [W547. University Course Platform: User Auth, Courses, Favorites API & DB Design](questions/W547.md)

## Foundation design diagrams

- [F039. Design a fair GPU job scheduler.](04_serving_and_training_systems.md#f039)
- [F067. Build an LLM evaluation harness.](07_evaluation_and_data.md#f067)
- [F071. RAG architecture: indexing, retrieval and grounded answers](08_rag_and_agents.md#f071)
- [F073. Hybrid search: lexical and semantic candidates, fusion and reranking](08_rag_and_agents.md#f073)
- [F076. Bounded agent loop: proposals, authorized tools and verified results](08_rag_and_agents.md#f076)
- [F077. Design tool schemas and validate LLM outputs.](08_rag_and_agents.md#f077)
- [F079. Trust boundary: retrieved text does not grant capabilities](08_rag_and_agents.md#f079)
- [F081. How to structure an ML system-design answer](09_ml_system_design.md#f081)
- [F082. Recommendation: candidates, ranking and feedback](09_ml_system_design.md#f082)
- [F083. Multimodal moderation: decisions and human review](09_ml_system_design.md#f083)
- [F084. Risk detection: event-time features and action policy](09_ml_system_design.md#f084)
- [F085. Image-text retrieval: compatible embeddings and ranked results](09_ml_system_design.md#f085)
- [F086. Feature platform: historical truth and online availability](09_ml_system_design.md#f086)
- [F087. Model rollout: shadow, canary, promotion and rollback](09_ml_system_design.md#f087)
- [F088. Evaluation platform: immutable runs, judgments and reports](09_ml_system_design.md#f088)
- [F089. Training-data mining: permitted logs, validation and clean labels](09_ml_system_design.md#f089)
- [F090. Rate-limited classification: durable work and validated results](09_ml_system_design.md#f090)
- [F161. Explain CAP during a real network partition.](17_distributed_systems_depth.md#f161)
- [F162. Design a durable work queue with retries and leases.](17_distributed_systems_depth.md#f162)
- [F163. Handle dependency-aware job scheduling.](17_distributed_systems_depth.md#f163)
- [F164. Design a read-once secret-sharing service.](17_distributed_systems_depth.md#f164)
- [F165. Design idempotency for an order or booking API.](17_distributed_systems_depth.md#f165)
- [F166. Design a token-bucket rate limiter.](17_distributed_systems_depth.md#f166)
- [F167. Design preemptible long-running generation jobs.](17_distributed_systems_depth.md#f167)
- [F168. Design a state-wide temperature ingestion and query system.](17_distributed_systems_depth.md#f168)
- [F169. Explain backpressure in a multi-stage data pipeline.](17_distributed_systems_depth.md#f169)
- [F170. Design a cache without serving incorrect data.](17_distributed_systems_depth.md#f170)
- [F177. Design an ETL job that can safely rerun.](18_data_and_sql.md#f177)
- [F181. Predict the category of a newly listed item.](19_product_ml_case_studies.md#f181)
- [F182. Recommend cover photos to improve bookings.](19_product_ml_case_studies.md#f182)
- [F183. Estimate listing lifetime value.](19_product_ml_case_studies.md#f183)
- [F184. Design personalized short-video ranking.](19_product_ml_case_studies.md#f184)
- [F185. Rank news without rewarding stale or misleading content.](19_product_ml_case_studies.md#f185)
- [F186. Design a hiring-workflow assistant without making unsupported automated decisions.](19_product_ml_case_studies.md#f186)
- [F187. Design a support-ticket routing model.](19_product_ml_case_studies.md#f187)
- [F188. Design a flight-search and booking agent.](19_product_ml_case_studies.md#f188)
- [F189. Detect possible image copyright violations as a retrieval problem.](19_product_ml_case_studies.md#f189)
- [F190. Classify viewers using movie completion behavior.](19_product_ml_case_studies.md#f190)
