const assert=require('node:assert/strict');
const fs=require('node:fs');
const {calculate}=require('../dist/app.js');
const data=JSON.parse(fs.readFileSync('results/evaluation.json','utf8'));
let checks=0;
for(const model of data.models)for(const split of ['validation','test']){
 const r=calculate(data.splits[split].y,model.results[split].p,model.threshold);
 for(const [key,value] of Object.entries(r)){assert.ok(Math.abs(value-model.results[split].metrics[key])<1e-12);checks++;}
 const zero=calculate(data.splits[split].y,model.results[split].p,0);
 assert.equal(zero.fn,0);assert.equal(zero.tn,0);checks+=2;
}
assert.equal(calculate([0,1],[.1,.9],1).precision,0);checks++;
console.log(`${checks} JavaScript metric checks passed`);
