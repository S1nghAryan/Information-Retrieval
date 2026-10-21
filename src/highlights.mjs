const stopwords = new Set('a an the and or but of to in on at by for from with as is are was were be been being it its this that these those i we you your they them their my our which what how why when where who can could should would will may might do does did have has had not no most more well also very than then there about into only any all some such'.split(' '));
const words = text => (text.toLowerCase().match(/[a-z]+(?:['’-][a-z]+)*/g) || []).filter(word => word.length > 2 && !stopwords.has(word));

// Return text segments, never HTML: source content stays escaped by React.
export function highlightSegments(text, question, evidence, rank) {
  const keywords = new Set(words(question + ' ' + evidence.map(item => item.quote).join(' ')));
  const ranges = [];
  for (const item of evidence.filter(item => item.source === rank)) {
    if (!item.quote) continue;
    let start = text.indexOf(item.quote);
    while (start !== -1) {
      ranges.push({start, end:start + item.quote.length, kind:'quote'});
      start = text.indexOf(item.quote, start + item.quote.length);
    }
  }
  for (const match of text.matchAll(/[a-z]+(?:['’-][a-z]+)*/gi)) {
    const start = match.index, end = start + match[0].length;
    if (keywords.has(match[0].toLowerCase()) && !ranges.some(range => start < range.end && end > range.start)) ranges.push({start, end, kind:'keyword'});
  }
  ranges.sort((a,b) => a.start-b.start || b.end-a.end);
  const parts = []; let cursor = 0;
  for (const range of ranges) {
    if (range.start < cursor) continue;
    if (range.start > cursor) parts.push({text:text.slice(cursor,range.start)});
    parts.push({text:text.slice(range.start,range.end),kind:range.kind});
    cursor=range.end;
  }
  if (cursor < text.length) parts.push({text:text.slice(cursor)});
  return parts;
}
