import pandas as pd
import numpy as np


def make_groups(similarity_file, threshold=0.75):

    similarity = pd.read_csv(
        similarity_file,
        index_col=0
    )

    components = list(similarity.index)
    n = len(components)

    # -----------------------------------------
    # Build connections between similar
    # components
    # -----------------------------------------

    graph = {
        i: set()
        for i in range(n)
    }

    for i in range(n):
        for j in range(i + 1, n):

            score = similarity.iloc[i, j]

            if score >= threshold:
                graph[i].add(j)
                graph[j].add(i)

    # -----------------------------------------
    # Find connected groups
    # -----------------------------------------

    visited = set()
    groups = []

    for start in range(n):

        if start in visited:
            continue

        stack = [start]
        group = []

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            group.append(current)

            for neighbor in graph[current]:

                if neighbor not in visited:
                    stack.append(neighbor)

        groups.append(sorted(group))

    # -----------------------------------------
    # Convert indexes to component names
    # -----------------------------------------

    named_groups = []

    for group in groups:

        named_group = [
            components[i]
            for i in group
        ]

        named_groups.append(named_group)

    return named_groups


groups = make_groups(
    "NMF/component_similarity.csv",
    threshold=0.8 
)

for i, group in enumerate(groups):

    print(f"Group {i}:")
    print(group)

print("-----------------------------------------------------")



def make_hard_groups(similarity_file, threshold=0.75):

    similarity = pd.read_csv(
        similarity_file,
        index_col=0
    )

    components = list(similarity.index)
    n = len(components)

    groups = []

    for i in range(n):

        # Try to place this component into an
        # existing group
        placed = False

        for group in groups:

            # It must be similar enough to EVERY
            # existing member of the group
            valid = all(
                similarity.iloc[i, j] >= threshold
                for j in group
            )

            if valid:
                group.append(i)
                placed = True
                break

        # No existing group satisfies the condition
        if not placed:
            groups.append([i])

    # Convert indexes to component names
    named_groups = []

    for group in groups:

        named_groups.append([
            components[i]
            for i in group
        ])

    return named_groups


groups = make_hard_groups(
    "nmf/component_similarity.csv",
    threshold=0.65
)

for i, group in enumerate(groups):
    print(f"Group {i}:")
    print(group)