import {readFileSync, writeFileSync} from 'node:fs';
import {createWindowDecoder} from '../frontend/tcn_stream.js';

const source=JSON.parse(readFileSync(new URL(process.argv[3]||'../data/stream-replay-scores.json',import.meta.url),'utf8'));
const groups={};
for(const row of source.cases) {
  const decoder=createWindowDecoder(), emitted=[];
  for(const observation of row.observations) {
    const result=decoder.update(observation,observation.now);
    if(result.commit) emitted.push({label:result.label,at:observation.now});
  }
  const group=groups[row.kind]??={total:0,exact:0,empty:0,extra:0,firstCommitMs:[],byLabel:{}};
  group.total++;
  const exact=JSON.stringify(emitted.map(item=>item.label))===JSON.stringify(row.expected);
  if(exact) group.exact++;
  if(!emitted.length) group.empty++;
  if(emitted.length>row.expected.length) group.extra++;
  if(emitted.length) group.firstCommitMs.push(emitted[0].at);
  const key=row.expected.join(' → ');
  const item=group.byLabel[key]??={total:0,exact:0};item.total++;if(exact)item.exact++;
}
for(const group of Object.values(groups)) {
  group.meanFirstCommitMs=group.firstCommitMs.length?Math.round(group.firstCommitMs.reduce((a,b)=>a+b,0)/group.firstCommitMs.length):null;
  delete group.firstCommitMs;
}
const report={note:source.note,groups};
const output=new URL(process.argv[2]||'../data/stream-replay-report.json',import.meta.url);
writeFileSync(output,JSON.stringify(report,null,2));
console.log(JSON.stringify({...report,groups:Object.fromEntries(Object.entries(groups).map(([key,{byLabel,...group}])=>[key,group]))}));
