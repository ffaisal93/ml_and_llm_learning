export function matches(item, { query = "", status = "all", type = "all" }) {
  const q = query.trim().toLocaleLowerCase();
  const textMatches =
    !q ||
    item.name.toLocaleLowerCase().includes(q) ||
    item.description.toLocaleLowerCase().includes(q);
  return (
    textMatches &&
    (status === "all" || item.status === status) &&
    (type === "all" || item.type === type)
  );
}
