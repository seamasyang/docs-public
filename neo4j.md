


### unit lesson and group



```
MATCH path = (u:Unit {name: "unit 01"})-[:HAS_LESSON]->(l:Lesson)-[:HAS_GROUP]->(g:Group)
RETURN path;
```

### unit和lession的关系



```
MATCH path = (u:Unit {name: "unit 01"})-[:HAS_LESSON]->(l:Lesson)
RETURN path;
```

查看整个 `lesson 01` 的完整结构

```
MATCH path = (l:Lesson {name: "lesson 01"})-[:HAS_GROUP]->(g:Group)-[:CONTAINS]->(w:Word)
RETURN path;
```
