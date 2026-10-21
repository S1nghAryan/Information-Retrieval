import React from 'react';

export default function MetricsCharts({rows, split, names, labels, colors, metric, onMetricChange}) {
  const baseline = rows.find(row => row.method === 'semantic_chunked');
  const differences = baseline ? rows.filter(row => row !== baseline).map(row => ({...row, difference: row.metrics[metric] - baseline.metrics[metric]})) : [];
  const extent = Math.max(.01, ...differences.map(row => Math.abs(row.difference)));
  const timed = rows.filter(row => Number.isFinite(row.metrics.mean_query_latency_ms) && row.metrics.mean_query_latency_ms >= 0);
  const maxTime = Math.max(1, ...timed.map(row => row.metrics.mean_query_latency_ms));
  const untimed = rows.filter(row => !timed.includes(row));
  return <section className="extra-charts" aria-label={`${split} evaluation charts`}>
    <article className="panel score-map">
      <div className="panel-title"><h2>Every score, at a glance.</h2><span className="pill">{split.toUpperCase()} · ALL METRICS</span></div>
      <p>Darker cells indicate higher scores on the same 0–1 scale. Select a metric heading to update the comparison charts.</p>
      <div className="table-scroll"><table className="heatmap">
        <caption className="sr-only">All retrieval methods and six evaluation scores for {split}</caption>
        <thead><tr><th scope="col">Retrieval method</th>{Object.entries(labels).map(([key, label]) => <th scope="col" key={key}><button aria-pressed={metric === key} onClick={() => onMetricChange(key)}>{label}</button></th>)}</tr></thead>
        <tbody>{rows.map(row => <tr key={row.method}><th scope="row">{names[row.method]}</th>{Object.entries(labels).map(([key, label]) => {
          const score = row.metrics[key];
          return <td key={key} style={{backgroundColor:`rgba(238,43,71,${.06 + score * .72})`, color:'#2C2E3E'}} title={`${names[row.method]} · ${label}: ${score.toFixed(6)}`}><strong>{score.toFixed(3)}</strong><span className="cell-meter" aria-hidden="true"><span style={{width:`${score * 100}%`}}/></span></td>;
        })}</tr>)}</tbody>
      </table></div>
      <div className="heat-legend"><span>0 · Lower</span><span className="heat-gradient" aria-hidden="true"/><span>1 · Higher</span></div>
    </article>
    <div className="chart-grid">
      <article className="panel difference-panel">
        <div className="panel-title"><h2>Compared with semantic.</h2><select aria-label="Difference chart metric" value={metric} onChange={event => onMetricChange(event.target.value)}>{Object.entries(labels).map(([key,label]) => <option value={key} key={key}>{label}</option>)}</select></div>
        <p>Absolute difference in {labels[metric]} from chunked semantic retrieval. Left is lower; right is higher.</p>
        <div className="delta-axis"><span>−{extent.toFixed(3)}</span><span>0</span><span>+{extent.toFixed(3)}</span></div>
        {differences.map(row => <div className="delta-row" key={row.method} title={`${names[row.method]} minus chunked semantic: ${row.difference.toFixed(6)}`}>
          <div className="chart-row-label"><span>{names[row.method]}</span><strong>{row.difference > 0 ? '+' : ''}{row.difference.toFixed(4)}</strong></div>
          <div className="delta-track" aria-hidden="true"><span className="delta-zero"/><span className={`delta-fill ${row.difference < 0 ? 'negative' : 'positive'}`} style={{left:row.difference < 0 ? `${50 - Math.abs(row.difference) / extent * 50}%` : '50%',width:`${Math.abs(row.difference) / extent * 50}%`}}/></div>
        </div>)}
        <p className="chart-note">Baseline: {baseline?.metrics[metric].toFixed(4)}. Differences are descriptive, not a statistical significance test. The methods use their documented saved configurations.</p>
      </article>
      <article className="panel latency-panel">
        <div className="panel-title"><h2>How long does retrieval take?</h2><span className="pill">LOWER IS FASTER</span></div>
        <p>Saved mean query latency in milliseconds. Model generation time is excluded.</p>
        <div className="latency-axis"><span>0 ms</span><span>{maxTime.toFixed(1)} ms</span></div>
        {timed.map(row => <div className="latency-row" key={row.method}>
          <div className="chart-row-label"><span>{names[row.method]}</span><strong>{row.metrics.mean_query_latency_ms.toFixed(2)} ms</strong></div>
          <div className="track" title={`${names[row.method]}: ${row.metrics.mean_query_latency_ms.toFixed(3)} ms`}><div className="bar" style={{width:`${row.metrics.mean_query_latency_ms / maxTime * 100}%`,background:colors[row.method]}}/></div>
        </div>)}
        {!!untimed.length && <div className="unmeasured"><strong>Not measured</strong><p>{untimed.map(row => names[row.method]).join(' · ')}</p></div>}
        <p className="chart-note">Missing timings are not zero. These saved runs were not a controlled repeated speed benchmark; loading and cache conditions may differ.</p>
      </article>
    </div>
  </section>;
}
