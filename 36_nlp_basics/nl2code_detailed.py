"""
NL2Code: Natural Language to Code Generation
Detailed implementation with schema handling
"""
from typing import List, Dict, Tuple
import re

class SchemaElement:
    """Represents a schema element (table, column, etc.)"""
    def __init__(self, name: str, element_type: str, description: str = ""):
        self.name = name
        self.element_type = element_type  # 'table', 'column', 'function', etc.
        self.description = description
    
    def __repr__(self):
        return f"{self.element_type}:{self.name}"

class Schema:
    """Database schema representation"""
    def __init__(self):
        self.tables = {}  # table_name -> SchemaElement
        self.columns = {}  # (table_name, column_name) -> SchemaElement
        self.relationships = []  # [(table1, table2, relationship_type)]
    
    def add_table(self, name: str, description: str = ""):
        """Add a table to schema"""
        self.tables[name] = SchemaElement(name, "table", description)
    
    def add_column(self, table: str, column: str, description: str = ""):
        """Add a column to schema"""
        key = (table, column)
        self.columns[key] = SchemaElement(column, "column", description)
    
    def add_relationship(self, table1: str, table2: str, rel_type: str = "foreign_key"):
        """Add relationship between tables"""
        self.relationships.append((table1, table2, rel_type))

class SchemaPruner:
    """
    Schema Pruning for Large Database Schemas
    
    Problem: Large schemas (thousands of tables/columns) don't fit in context
    Solution: Select only relevant schema elements based on query
    """
    
    def __init__(self, schema: Schema):
        self.schema = schema
    
    def compute_relevance_score(self, query: str, element: SchemaElement) -> float:
        """
        Compute relevance score between query and schema element
        
        Methods:
        1. Keyword matching (TF-IDF)
        2. Embedding similarity (BERT)
        3. Description matching
        """
        # Simple keyword-based scoring
        query_lower = query.lower()
        element_lower = element.name.lower()
        desc_lower = element.description.lower()
        
        score = 0.0
        
        # Exact match
        if element_lower in query_lower or query_lower in element_lower:
            score += 10.0
        
        # Word overlap
        query_words = set(query_lower.split())
        element_words = set(element_lower.split())
        overlap = len(query_words & element_words)
        score += overlap * 2.0
        
        # Description match
        if desc_lower:
            desc_words = set(desc_lower.split())
            desc_overlap = len(query_words & desc_words)
            score += desc_overlap * 1.0
        
        return score
    
    def prune_schema(self, query: str, top_k_tables: int = 5, 
                    top_k_columns_per_table: int = 10) -> Schema:
        """
        Prune schema to only relevant elements
        
        Steps:
        1. Score all tables by relevance
        2. Select top-K tables
        3. For each table, select top-K columns
        4. Include relationships between selected tables
        """
        # Score tables
        table_scores = []
        for table_name, table_element in self.schema.tables.items():
            score = self.compute_relevance_score(query, table_element)
            table_scores.append((table_name, score))
        
        # Select top-K tables
        table_scores.sort(key=lambda x: x[1], reverse=True)
        selected_tables = [name for name, _ in table_scores[:top_k_tables]]
        
        # Create pruned schema
        pruned = Schema()
        
        # Add selected tables
        for table_name in selected_tables:
            table_element = self.schema.tables[table_name]
            pruned.add_table(table_name, table_element.description)
        
        # Add relevant columns for selected tables
        for table_name in selected_tables:
            # Score columns for this table
            column_scores = []
            for (t, c), col_element in self.schema.columns.items():
                if t == table_name:
                    score = self.compute_relevance_score(query, col_element)
                    column_scores.append((c, score))
            
            # Select top-K columns
            column_scores.sort(key=lambda x: x[1], reverse=True)
            selected_columns = [name for name, _ in column_scores[:top_k_columns_per_table]]
            
            # Add columns
            for col_name in selected_columns:
                key = (table_name, col_name)
                col_element = self.schema.columns[key]
                pruned.add_column(table_name, col_name, col_element.description)
        
        # Add relationships between selected tables
        for t1, t2, rel_type in self.schema.relationships:
            if t1 in selected_tables and t2 in selected_tables:
                pruned.add_relationship(t1, t2, rel_type)
        
        return pruned
    
    def hierarchical_pruning(self, query: str, max_elements: int = 50) -> Schema:
        """
        Hierarchical pruning: Prune at different levels
        
        Strategy:
        1. First select most relevant tables
        2. Then select most relevant columns from those tables
        3. Ensure total elements <= max_elements
        """
        # Score all elements
        all_scores = []
        
        # Table scores
        for table_name, table_element in self.schema.tables.items():
            score = self.compute_relevance_score(query, table_element)
            all_scores.append(('table', table_name, score))
        
        # Column scores
        for (table, column), col_element in self.schema.columns.items():
            score = self.compute_relevance_score(query, col_element)
            all_scores.append(('column', (table, column), score))
        
        # Sort by score
        all_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Build pruned schema
        pruned = Schema()
        added_tables = set()
        added_columns = set()
        
        for element_type, element_id, score in all_scores[:max_elements]:
            if element_type == 'table':
                table_name = element_id
                if table_name not in added_tables:
                    table_element = self.schema.tables[table_name]
                    pruned.add_table(table_name, table_element.description)
                    added_tables.add(table_name)
            
            elif element_type == 'column':
                table, column = element_id
                if table in added_tables and (table, column) not in added_columns:
                    col_element = self.schema.columns[(table, column)]
                    pruned.add_column(table, column, col_element.description)
                    added_columns.add((table, column))
        
        return pruned

class NL2CodeGenerator:
    """
    Natural Language to Code Generator
    
    Simplified version showing the pipeline
    """
    
    def __init__(self, schema: Schema):
        self.schema = schema
        self.pruner = SchemaPruner(schema)
    
    def generate_code(self, query: str, use_pruning: bool = True) -> str:
        """
        Generate code from natural language query
        
        Pipeline:
        1. Schema pruning (if needed)
        2. Schema encoding
        3. Code generation
        """
        # Step 1: Schema pruning (for large schemas)
        if use_pruning:
            pruned_schema = self.pruner.prune_schema(query, top_k_tables=5)
        else:
            pruned_schema = self.schema
        
        # Step 2: Schema encoding (simplified)
        schema_context = self._encode_schema(pruned_schema)
        
        # Step 3: Code generation (simplified - would use actual model)
        code = self._generate_from_query(query, schema_context)
        
        return code
    
    def _encode_schema(self, schema: Schema) -> str:
        """Encode schema into text format"""
        context_parts = []
        
        # Encode tables
        for table_name, table_element in schema.tables.items():
            table_info = f"Table: {table_name}"
            if table_element.description:
                table_info += f" ({table_element.description})"
            context_parts.append(table_info)
            
            # Encode columns for this table
            for (t, c), col_element in schema.columns.items():
                if t == table_name:
                    col_info = f"  - {c}"
                    if col_element.description:
                        col_info += f": {col_element.description}"
                    context_parts.append(col_info)
        
        # Encode relationships
        for t1, t2, rel_type in schema.relationships:
            context_parts.append(f"Relationship: {t1} -> {t2} ({rel_type})")
        
        return "\n".join(context_parts)
    
    def _generate_from_query(self, query: str, schema_context: str) -> str:
        """
        Generate code from query and schema context
        
        In practice, this would use a fine-tuned language model
        (e.g., CodeT5, StarCoder, GPT-3.5 Code)
        """
        # Simplified rule-based generation (for demonstration)
        # In practice, use a trained model
        
        query_lower = query.lower()
        
        # Simple pattern matching (would be replaced with actual model)
        if "select" in query_lower or "find" in query_lower:
            # Generate SELECT query
            tables = list(self.schema.tables.keys())
            if tables:
                table = tables[0]
                columns = [c for (t, c) in self.schema.columns.keys() if t == table]
                
                if columns:
                    cols_str = ", ".join(columns[:3])  # Limit columns
                    return f"SELECT {cols_str}\nFROM {table};"
        
        return "-- Generated code placeholder\n-- Would use actual model here"

# Example Usage
if __name__ == "__main__":
    print("NL2Code: Natural Language to Code")
    print("=" * 60)
    
    # Create large schema
    schema = Schema()
    
    # Add many tables (simulating large schema)
    for i in range(100):
        schema.add_table(f"table_{i}", f"Table number {i}")
        for j in range(20):
            schema.add_column(f"table_{i}", f"col_{j}", f"Column {j} of table {i}")
    
    # Add relevant tables for our query
    schema.add_table("customers", "Customer information")
    schema.add_column("customers", "id", "Customer ID")
    schema.add_column("customers", "name", "Customer name")
    schema.add_column("customers", "email", "Customer email")
    
    schema.add_table("orders", "Order information")
    schema.add_column("orders", "id", "Order ID")
    schema.add_column("orders", "customer_id", "Customer who placed order")
    schema.add_column("orders", "date", "Order date")
    schema.add_column("orders", "total", "Order total amount")
    
    schema.add_relationship("orders", "customers", "foreign_key")
    
    print(f"Original schema: {len(schema.tables)} tables, {len(schema.columns)} columns")
    print()
    
    # Query
    query = "Find all customers who placed orders in 2023"
    
    # Prune schema
    pruner = SchemaPruner(schema)
    pruned = pruner.prune_schema(query, top_k_tables=5, top_k_columns_per_table=10)
    
    print(f"Pruned schema: {len(pruned.tables)} tables, {len(pruned.columns)} columns")
    print("\nSelected tables:")
    for table in pruned.tables:
        print(f"  - {table}")
    print()
    
    # Generate code
    generator = NL2CodeGenerator(schema)
    code = generator.generate_code(query, use_pruning=True)
    
    print("Generated code:")
    print(code)
    print()
    
    print("Key Points:")
    print("  1. Schema pruning reduces context size")
    print("  2. Relevance scoring selects important elements")
    print("  3. Hierarchical pruning can be used for very large schemas")
    print("  4. In practice, use fine-tuned models (CodeT5, StarCoder)")

