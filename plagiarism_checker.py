from pathlib import Path

DEFAULT_SHINGLE_SIZE = 3
HIGH_SIMILARITY_THRESHOLD = 0.75
MODERATE_SIMILARITY_THRESHOLD = 0.40


class HashEntry:
    # One item stored inside the hash table.
    def __init__(self, key, value):
        self.key = key
        self.value = value


class CustomHashTable:
    def __init__(self, initial_capacity=11):
        # Start with a small table and resize later if needed.
        self.capacity = initial_capacity
        self.size = 0
        self.buckets = []

        # Each bucket starts as an empty list for collision handling.
        for i in range(self.capacity):
            self.buckets.append([])

    def _hash(self, key):
        key_text = str(key)
        hash_value = 0

        # Basic polynomial hash function.
        for character in key_text:
            hash_value = (hash_value * 31 + ord(character)) % self.capacity

        return hash_value

    def _find_entry(self, key):
        # Search only the bucket where this key should be.
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]

        for entry in bucket:
            if entry.key == key:
                return entry

        return None

    def _resize_if_needed(self):
        load_factor = self.size / self.capacity

        # Keep the table from getting too crowded.
        if load_factor <= 0.70:
            return

        old_buckets = self.buckets
        self.capacity = self.capacity * 2 + 1
        self.size = 0
        self.buckets = []

        for i in range(self.capacity):
            self.buckets.append([])

        # Reinsert old entries because the bucket indexes change.
        for bucket in old_buckets:
            for entry in bucket:
                self.insert(entry.key, entry.value)

    def insert(self, key, value):
        # If the key already exists, replace its old value.
        existing_entry = self._find_entry(key)

        if existing_entry is not None:
            existing_entry.value = value
            return

        self._resize_if_needed()
        bucket_index = self._hash(key)

        # Collisions are handled by keeping a small list at each bucket.
        self.buckets[bucket_index].append(HashEntry(key, value))
        self.size += 1

    def get(self, key, default_value=None):
        # Return a default value if the key is not in the table.
        entry = self._find_entry(key)

        if entry is None:
            return default_value

        return entry.value

    def contains(self, key):
        # Used when we only need to know if a key exists.
        return self._find_entry(key) is not None

    def increment(self, key):
        entry = self._find_entry(key)

        if entry is None:
            # First time seeing this key.
            self.insert(key, 1)
        else:
            entry.value += 1

    def entries(self):
        # Lets other code scan the table without building a new list.
        for bucket in self.buckets:
            for entry in bucket:
                yield entry

    def count(self):
        # Number of different keys stored in the table.
        return self.size


class Document:
    # Stores the basic information we need for each file.
    def __init__(self, document_id, file_name, file_path, raw_text):
        self.document_id = document_id
        self.file_name = file_name
        self.file_path = file_path
        self.raw_text = raw_text
        self.tokens = []
        self.token_frequencies = None
        self.shingles = None

    def display_name(self):
        # Short name used in terminal output.
        return "D" + str(self.document_id) + " - " + self.file_name


class SimilarityResult:
    # Stores the similarity score for one pair of documents.
    def __init__(self, first_document, second_document, score, shared_count, total_count):
        self.first_document = first_document
        self.second_document = second_document
        self.score = score
        self.shared_count = shared_count
        self.total_count = total_count
        self.label = ""
        self.is_suspicious = False


class BSTNode:
    # One node in the suspicious-pair tree.
    def __init__(self, result):
        self.key_score = result.score
        self.key_shared_count = result.shared_count
        self.value = result
        self.left = None
        self.right = None


class SuspiciousPairBST:
    def __init__(self):
        # Stores suspicious pairs ordered by similarity.
        self.root = None
        self.size = 0

    def insert(self, result):
        # Only suspicious pairs belong in this tree.
        if not result.is_suspicious:
            return

        self.root = self._insert_recursive(self.root, result)
        self.size += 1

    def _insert_recursive(self, current_node, result):
        if current_node is None:
            return BSTNode(result)

        comparison = self._compare_result_to_node(result, current_node)

        if comparison < 0:
            current_node.left = self._insert_recursive(current_node.left, result)
        else:
            current_node.right = self._insert_recursive(current_node.right, result)

        return current_node

    def _compare_result_to_node(self, result, current_node):
        # Compare against the key values stored in the current node.
        if result.score > current_node.key_score:
            return -1

        if result.score < current_node.key_score:
            return 1

        if result.shared_count > current_node.key_shared_count:
            return -1

        if result.shared_count < current_node.key_shared_count:
            return 1

        return compare_similarity_results(result, current_node.value)

    def in_order_traversal(self):
        # Left-root-right gives highest similarity first with our ordering.
        ordered_results = []
        self._in_order_recursive(self.root, ordered_results)
        return ordered_results

    def _in_order_recursive(self, current_node, ordered_results):
        if current_node is None:
            return

        self._in_order_recursive(current_node.left, ordered_results)
        ordered_results.append(current_node.value)
        self._in_order_recursive(current_node.right, ordered_results)

    def count(self):
        return self.size


def read_text_documents(folder_path):
    folder = Path(folder_path)

    # Check the folder before trying to read from it.
    if not folder.exists():
        raise FileNotFoundError("Folder does not exist: " + str(folder))

    if not folder.is_dir():
        raise NotADirectoryError("Path is not a folder: " + str(folder))

    documents = []
    next_document_id = 1

    for entry in folder.iterdir():
        # Only text files are part of the document comparison.
        if entry.is_file() and entry.suffix.lower() == ".txt":
            raw_text = entry.read_text(encoding="utf-8", errors="replace")
            document = Document(
                next_document_id,
                entry.name,
                str(entry),
                raw_text
            )
            documents.append(document)
            next_document_id += 1

    # The rest of the program needs at least one document to work with.
    if len(documents) == 0:
        raise ValueError("No .txt files were found in: " + str(folder))

    return documents


def preprocess_text(raw_text):
    # Turn raw document text into clean word tokens.
    lower_text = raw_text.lower()
    cleaned_characters = []

    for character in lower_text:
        if character.isalnum() or character.isspace():
            cleaned_characters.append(character)
        else:
            # Use a space so two words do not accidentally join together.
            cleaned_characters.append(" ")

    cleaned_text = "".join(cleaned_characters)
    tokens = cleaned_text.split()
    return tokens


def preprocess_documents(documents):
    # Save the cleaned tokens inside each document object.
    for document in documents:
        document.tokens = preprocess_text(document.raw_text)


def build_token_frequency_table(tokens):
    # Counts how many times each word appears using our hash table.
    frequency_table = CustomHashTable()

    for token in tokens:
        frequency_table.increment(token)

    return frequency_table


def build_document_token_tables(documents):
    # Build one frequency table for each document.
    for document in documents:
        document.token_frequencies = build_token_frequency_table(document.tokens)


def create_shingle(tokens, start_index, shingle_size):
    # Builds one phrase from neighboring words.
    shingle_words = []
    end_index = start_index + shingle_size

    for index in range(start_index, end_index):
        shingle_words.append(tokens[index])

    return " ".join(shingle_words)


def build_shingle_table(tokens, shingle_size=DEFAULT_SHINGLE_SIZE):
    # Stores shingles in our hash table instead of using a set.
    shingle_table = CustomHashTable()

    if len(tokens) == 0:
        return shingle_table

    if len(tokens) < shingle_size:
        short_shingle = " ".join(tokens)
        shingle_table.increment(short_shingle)
        return shingle_table

    last_start_index = len(tokens) - shingle_size

    for start_index in range(last_start_index + 1):
        shingle = create_shingle(tokens, start_index, shingle_size)
        shingle_table.increment(shingle)

    return shingle_table


def build_document_shingle_tables(documents, shingle_size=DEFAULT_SHINGLE_SIZE):
    # Build one shingle table for each document.
    for document in documents:
        document.shingles = build_shingle_table(document.tokens, shingle_size)


def calculate_jaccard_similarity(first_table, second_table):
    # Jaccard compares shared unique shingles against total unique shingles.
    if first_table.count() <= second_table.count():
        smaller_table = first_table
        larger_table = second_table
    else:
        smaller_table = second_table
        larger_table = first_table

    shared_count = 0

    for entry in smaller_table.entries():
        if larger_table.contains(entry.key):
            shared_count += 1

    total_count = first_table.count() + second_table.count() - shared_count

    if total_count == 0:
        return 0.0, shared_count, total_count

    score = shared_count / total_count
    return score, shared_count, total_count


def compare_document_pair(first_document, second_document):
    # Compare two documents using their shingle hash tables.
    score, shared_count, total_count = calculate_jaccard_similarity(
        first_document.shingles,
        second_document.shingles
    )

    return SimilarityResult(
        first_document,
        second_document,
        score,
        shared_count,
        total_count
    )


def compare_all_document_pairs(documents):
    # Check every document pair exactly once.
    results = []

    for first_index in range(len(documents)):
        for second_index in range(first_index + 1, len(documents)):
            result = compare_document_pair(
                documents[first_index],
                documents[second_index]
            )
            results.append(result)

    return results


def classify_similarity_score(score):
    # Simple thresholds for deciding how suspicious a pair is.
    if score >= HIGH_SIMILARITY_THRESHOLD:
        return "High"

    if score >= MODERATE_SIMILARITY_THRESHOLD:
        return "Moderate"

    return "Low"


def is_suspicious_label(label):
    # High and Moderate pairs are worth showing as suspicious.
    return label == "High" or label == "Moderate"


def classify_similarity_results(results):
    # Add a label to every pair result.
    for result in results:
        result.label = classify_similarity_score(result.score)
        result.is_suspicious = is_suspicious_label(result.label)


def compare_similarity_results(first_result, second_result):
    # Returns -1 if the first result should be ranked before the second.
    if first_result.score > second_result.score:
        return -1

    if first_result.score < second_result.score:
        return 1

    # Tie-breaker: more shared shingles is a bit more suspicious.
    if first_result.shared_count > second_result.shared_count:
        return -1

    if first_result.shared_count < second_result.shared_count:
        return 1

    if first_result.first_document.document_id < second_result.first_document.document_id:
        return -1

    if first_result.first_document.document_id > second_result.first_document.document_id:
        return 1

    if first_result.second_document.document_id < second_result.second_document.document_id:
        return -1

    if first_result.second_document.document_id > second_result.second_document.document_id:
        return 1

    return 0


def should_come_before(first_result, second_result):
    # Used by merge sort to keep results in descending similarity order.
    return compare_similarity_results(first_result, second_result) <= 0


def merge_sorted_result_lists(left_results, right_results):
    # Merge two already-sorted halves into one sorted list.
    merged_results = []
    left_index = 0
    right_index = 0

    while left_index < len(left_results) and right_index < len(right_results):
        if should_come_before(left_results[left_index], right_results[right_index]):
            merged_results.append(left_results[left_index])
            left_index += 1
        else:
            merged_results.append(right_results[right_index])
            right_index += 1

    while left_index < len(left_results):
        merged_results.append(left_results[left_index])
        left_index += 1

    while right_index < len(right_results):
        merged_results.append(right_results[right_index])
        right_index += 1

    return merged_results


def merge_sort_similarity_results(results):
    # Divide the list, sort each half, then merge them back together.
    if len(results) <= 1:
        return results

    middle_index = len(results) // 2
    left_half = []
    right_half = []

    for index in range(0, middle_index):
        left_half.append(results[index])

    for index in range(middle_index, len(results)):
        right_half.append(results[index])

    sorted_left = merge_sort_similarity_results(left_half)
    sorted_right = merge_sort_similarity_results(right_half)
    return merge_sorted_result_lists(sorted_left, sorted_right)


def build_suspicious_pair_tree(results):
    # Store only suspicious pairs in a BST for ordered retrieval.
    suspicious_tree = SuspiciousPairBST()

    for result in results:
        suspicious_tree.insert(result)

    return suspicious_tree


def print_report_line():
    # Keeps the terminal output easier to read.
    print("-" * 70)


def format_score(score):
    # Convert a decimal score into a percentage string.
    return format(score * 100, ".2f") + "%"


def print_report_header():
    # Main title for the final output.
    print()
    print_report_line()
    print("DOCUMENT SIMILARITY REPORT")
    print_report_line()
    print("Shingle size: " + str(DEFAULT_SHINGLE_SIZE))
    print("High threshold: " + format_score(HIGH_SIMILARITY_THRESHOLD))
    print("Moderate threshold: " + format_score(MODERATE_SIMILARITY_THRESHOLD))


def print_document_overview(documents):
    # Shows the documents that were included in the comparison.
    print()
    print("Documents analyzed:")

    for document in documents:
        print(
            document.display_name()
            + " | words: "
            + str(len(document.tokens))
            + " | unique words: "
            + str(document.token_frequencies.count())
            + " | unique shingles: "
            + str(document.shingles.count())
        )


def print_similarity_summary(results):
    # Shows document pairs after our custom merge sort.
    print()
    print("Ranked document pairs:")

    if len(results) == 0:
        print("At least two documents are needed for comparison.")
        return

    rank_number = 1

    for result in results:
        print(
            str(rank_number)
            + ". "
            + result.first_document.display_name()
            + " vs "
            + result.second_document.display_name()
        )
        print(
            "   Score: "
            + format_score(result.score)
            + " | Label: "
            + result.label
        )
        print(
            "   Shared shingles: "
            + str(result.shared_count)
            + " shared out of "
            + str(result.total_count)
            + " unique shingles"
        )
        rank_number += 1


def print_suspicious_pair_summary(suspicious_tree):
    # Shows suspicious pairs using the BST's ordered traversal.
    print()
    print("Suspicious pairs:")

    if suspicious_tree.count() == 0:
        print("No suspicious pairs found with the current thresholds.")
        return

    suspicious_results = suspicious_tree.in_order_traversal()
    rank_number = 1

    for result in suspicious_results:
        print(
            str(rank_number)
            + ". "
            + result.first_document.display_name()
            + " vs "
            + result.second_document.display_name()
            + " | "
            + format_score(result.score)
            + " | "
            + result.label
        )
        rank_number += 1


def print_final_report(documents, ranked_results, suspicious_tree):
    # Combines all final output in one place.
    print_report_header()
    print_document_overview(documents)
    print_similarity_summary(ranked_results)
    print_suspicious_pair_summary(suspicious_tree)
    print_report_line()


def main():
    # Main program flow.
    folder_path = input("Enter folder path containing .txt files: ").strip()
    documents = read_text_documents(folder_path)
    preprocess_documents(documents)
    build_document_token_tables(documents)
    build_document_shingle_tables(documents)
    similarity_results = compare_all_document_pairs(documents)
    classify_similarity_results(similarity_results)
    suspicious_pair_tree = build_suspicious_pair_tree(similarity_results)
    ranked_similarity_results = merge_sort_similarity_results(similarity_results)
    print_final_report(documents, ranked_similarity_results, suspicious_pair_tree)


if __name__ == "__main__":
    main()