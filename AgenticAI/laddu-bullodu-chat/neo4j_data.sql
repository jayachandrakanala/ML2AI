// ==========================================
// 1. CLEANUP & SCHEMA CONSTRAINTS
// ==========================================
MATCH (n) DETACH DELETE n;

CREATE CONSTRAINT unique_port_id IF NOT EXISTS FOR (p:Port) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT unique_supplier_id IF NOT EXISTS FOR (s:Supplier) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT unique_component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT unique_product_id IF NOT EXISTS FOR (prod:Product) REQUIRE prod.id IS UNIQUE;
CREATE CONSTRAINT unique_facility_id IF NOT EXISTS FOR (f:Facility) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT unique_warehouse_id IF NOT EXISTS FOR (w:Warehouse) REQUIRE w.id IS UNIQUE;
CREATE CONSTRAINT unique_order_id IF NOT EXISTS FOR (o:Order) REQUIRE o.id IS UNIQUE;

CREATE INDEX port_status_idx IF NOT EXISTS FOR (p:Port) ON (p.status);
CREATE INDEX order_status_idx IF NOT EXISTS FOR (o:Order) ON (o.status);

// ==========================================
// 2. CREATE NODES & NETWORK RELATIONSHIPS
// ==========================================
// Ports & Transit Hubs
CREATE (portLA:Port {id: 'PORT-LA', name: 'Port of Los Angeles', status: 'CONGESTED', location: 'California, US'})
CREATE (portSZ:Port {id: 'PORT-SZ', name: 'Port of Shenzhen', status: 'OPERATIONAL', location: 'Guangdong, CN'})
CREATE (portOAK:Port {id: 'PORT-OAK', name: 'Port of Oakland', status: 'OPERATIONAL', location: 'California, US'})

// Tier 2 Raw Material Suppliers
CREATE (s1:Supplier {id: 'SUP-01', name: 'Silicon Core Microchips', tier: 2, location: 'Shenzhen, CN'})
CREATE (s2:Supplier {id: 'SUP-02', name: 'Precision Lens Corp', tier: 2, location: 'Taiwan, TW'})

// Components & Products (Bill of Materials)
CREATE (c1:Component {id: 'CMP-101', name: 'A16 Microcontroller', stockLevel: 1500})
CREATE (c2:Component {id: 'CMP-202', name: '4K Camera Sensor Module', stockLevel: 300})
CREATE (p1:Product {id: 'PRD-900', name: 'Enterprise Security Camera Pro', retailPrice: 450})

// Manufacturing Facilities & Distribution Centers
CREATE (factory:Facility {id: 'FAC-US', name: 'Texas Assembly Plant', city: 'Austin', state: 'TX'})
CREATE (dc:Warehouse {id: 'DC-CENTRAL', name: 'Midwest Fulfillment Hub', city: 'Chicago', state: 'IL'})

// High-Value Customer Orders
CREATE (o1:Order {id: 'ORD-8801', customer: 'Acme Retailers', orderValue: 250000, status: 'PROCESSING', dueDate: '2026-11-15'})
CREATE (o2:Order {id: 'ORD-8802', customer: 'Global Defense Systems', orderValue: 850000, status: 'PROCESSING', dueDate: '2026-11-20'})

// Supply Chain & Routing Relationships (with weighted transit properties)
CREATE (s1)-[:SHIPS_VIA {transitDays: 14, carrier: 'Pacific Cargo'}]->(portSZ)
CREATE (portSZ)-[:ROUTES_TO {transitDays: 12, seaRoute: 'Transpacific East'}]->(portLA)
CREATE (portSZ)-[:ALTERNATE_ROUTE {transitDays: 16, addedLeadTimeDays: 4, costMultiplier: 1.12}]->(portOAK)

CREATE (portLA)-[:DELIVERS_TO {transitDays: 3, transportMode: 'Freight Rail'}]->(factory)
CREATE (portOAK)-[:DELIVERS_TO {transitDays: 4, transportMode: 'Freight Rail'}]->(factory)

CREATE (s1)-[:SUPPLIES {partNumber: 'A16-MCU'}]->(c1)
CREATE (s2)-[:SUPPLIES {partNumber: '4K-SENS'}]->(c2)

CREATE (c1)-[:REQUIRED_FOR {qtyPerUnit: 2}]->(p1)
CREATE (c2)-[:REQUIRED_FOR {qtyPerUnit: 1}]->(p1)

CREATE (factory)-[:MANUFACTURES]->(p1)
CREATE (p1)-[:STORED_AT]->(dc)
CREATE (dc)-[:FULFILLS]->(o1)
CREATE (dc)-[:FULFILLS]->(o2);

// ==========================================
// 3. GDS GRAPH PROJECTION CREATION
// ==========================================
// Drop projection if it already exists from a prior test run
CALL gds.graph.drop('supplyChainNetwork', false) YIELD graphName;

// Create fresh in-memory graph projection using 'properties' instead of 'property'
CALL gds.graph.project(
    'supplyChainNetwork',
    {
      Port: {label: 'Port'},
      Supplier: {label: 'Supplier'},
      Facility: {label: 'Facility'},
      Warehouse: {label: 'Warehouse'}
    },
    {
      ROUTES_TO: {type: 'ROUTES_TO', properties: ['transitDays'], orientation: 'NATURAL'},
      DELIVERS_TO: {type: 'DELIVERS_TO', properties: ['transitDays'], orientation: 'NATURAL'},
      ALTERNATE_ROUTE: {type: 'ALTERNATE_ROUTE', properties: ['transitDays'], orientation: 'NATURAL'},
      SHIPS_VIA: {type: 'SHIPS_VIA', properties: ['transitDays'], orientation: 'NATURAL'}
    }
);

// Ensure port status is explicitly set
MATCH (p:Port {id: 'PORT-LA'})
SET p.status = 'CONGESTED';

// Ensure compositeWeight is calculated and stored on relationships
MATCH ()-[r]->()
WHERE r.transitDays IS NOT NULL
SET r.compositeWeight = toFloat(r.transitDays) + (coalesce(r.costMultiplier, 1.0) * 4.0);