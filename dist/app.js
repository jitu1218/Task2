/* All metrics are recomputed from genuine held-out predictions, never invented. */
'use strict';
function calculate(y, p, threshold) {
  let tn=0,fp=0,fn=0,tp=0;
  y.forEach((label,i)=>{const pred=p[i]>=threshold;if(label===1){if(pred)tp++;else fn++;}else{if(pred)fp++;else tn++;}});
  return {tn,fp,fn,tp,precision:tp/(tp+fp)||0,recall:tp/(tp+fn)||0,f1:2*tp/(2*tp+fp+fn)||0,cost:fp+5*fn};
}
if(typeof module!=='undefined') module.exports={calculate};
if(typeof window!=='undefined') {
const data=window.EVALUATION;
let thresholds=Object.fromEntries(data.models.map(m=>[m.id,m.threshold]));
const $=id=>document.getElementById(id), fixed=n=>n.toFixed(3);
function current(m){const split=$('split').value;return {...calculate(data.splits[split].y,m.results[split].p,thresholds[m.id]),auc:m.results[split].metrics.auc,ap:m.results[split].metrics.ap};}
function render(){
 const split=$('split').value, selected=data.models.find(m=>m.id===$('model').value), n=data.splits[split].n;
 $('threshold').value=thresholds[selected.id];$('threshold-value').textContent=thresholds[selected.id].toFixed(2);
 const changed=data.models.some(m=>Math.abs(thresholds[m.id]-m.threshold)>1e-8);
 $('context').textContent=changed?'EXPLORATORY THRESHOLDS · Changes describe this split only. Do not select or tune a deployment policy on the test set.':'LOCKED BASELINE · Thresholds and model were selected on validation data. Adjust the slider to explore trade-offs.';
 $('context').className='context'+(changed?' changed':'');
 const a=current(selected);
 $('kpis').innerHTML=[['Malignant recall',`${(a.recall*100).toFixed(1)}%`,`${a.tp} of ${a.tp+a.fn} malignant cases detected`],['Precision',`${(a.precision*100).toFixed(1)}%`,`${a.tp} true positives / ${a.tp+a.fp} positive predictions`],['Missed cases',a.fn,`${selected.name} · ${split}`],['False alarms',a.fp,`${n} evaluated cases · threshold ${thresholds[selected.id].toFixed(2)}`]].map(([title,value,note])=>`<div class="kpi"><span>${title}</span><strong>${value}</strong><small>${note}</small></div>`).join('');
 $('metrics').innerHTML=data.models.map(m=>{const r=current(m);return `<tr class="${m.id===selected.id?'active':''}"><td><i class="dot ${m.id==='lr'?'teal':'orange'}"></i>${m.name}</td><td>${thresholds[m.id].toFixed(2)}</td><td>${fixed(r.precision)}</td><td>${fixed(r.recall)}</td><td>${fixed(r.f1)}</td><td>${fixed(r.auc)}</td><td>${fixed(r.ap)}</td><td>${r.cost}</td></tr>`;}).join('');
 $('matrices').innerHTML=data.models.map(m=>{const r=current(m);const cell=(v,label,error)=>`<div class="cell" style="background:${error?`rgba(217,119,64,${.08+.65*v/n})`:`rgba(0,127,120,${.08+.45*v/n})`}"><b>${v}</b><span>${label}</span></div>`;return `<div class="matrix-block"><div class="matrix-head"><strong>${m.name}</strong><span>t = ${thresholds[m.id].toFixed(2)}</span></div><div class="matrix"><div></div><div class="label">Predicted benign</div><div class="label">Predicted malignant</div><div class="label">Actual benign</div>${cell(r.tn,'True negative',false)}${cell(r.fp,'False positive',true)}<div class="label">Actual malignant</div>${cell(r.fn,'False negative',true)}${cell(r.tp,'True positive',false)}</div></div>`;}).join('');
 const sx=x=>60+390*x,sy=y=>310-270*y;
 let svg='<svg class="roc-svg" viewBox="0 0 490 360" role="img" aria-label="ROC curves comparing both models. See metrics table for AUC values.">';
 for(let i=0;i<=5;i++){const f=i/5;svg+=`<path d="M ${sx(f)} 40 V 310 M 60 ${sy(f)} H 450" stroke="#e3eae6" fill="none"/><text x="${sx(f)}" y="329" text-anchor="middle" font-size="10" fill="#627776">${f.toFixed(1)}</text><text x="45" y="${sy(f)+4}" text-anchor="end" font-size="10" fill="#627776">${f.toFixed(1)}</text>`;}
 svg+='<path d="M60 310L450 40" stroke="#afbbb3" stroke-dasharray="5 5"/><text x="255" y="352" text-anchor="middle" font-size="11" fill="#627776">False positive rate</text><text transform="translate(15 180) rotate(-90)" text-anchor="middle" font-size="11" fill="#627776">True positive rate / recall</text>';
 for(const m of data.models){const r=m.results[split],color=m.id==='lr'?'#007f78':'#d97740';svg+=`<path d="${r.fpr.map((x,i)=>`${i?'L':'M'}${sx(x)} ${sy(r.tpr[i])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5"/>`;const a=current(m),fpr=a.fp/(a.fp+a.tn);svg+=`<circle cx="${sx(fpr)}" cy="${sy(a.recall)}" r="5" fill="${color}" stroke="white" stroke-width="1.5"><title>${m.name}: FPR ${fixed(fpr)}, recall ${fixed(a.recall)}</title></circle>`;}
 $('roc').innerHTML=svg+'</svg>';
 const winner=data.models.find(m=>m.id===data.selected), v=winner.results.validation.metrics,t=winner.results.test.metrics;
 $('recommendation-title').textContent=`Validation preference: ${winner.name}`;
 $('recommendation').textContent=`At its frozen threshold ${winner.threshold.toFixed(2)}, this model had validation cost ${v.cost}. On the held-out test set it produced ${t.fn} missed malignant cases and ${t.fp} false alarms (recall ${(t.recall*100).toFixed(1)}%, precision ${(t.precision*100).toFixed(1)}%). This preference follows the declared cost policy, not whichever score looks best after moving a test slider. Small-sample results require further validation.`;
}
$('threshold').addEventListener('input',()=>{thresholds[$('model').value]=Number($('threshold').value);render();});
$('model').addEventListener('change',render);$('split').addEventListener('change',render);
$('reset').addEventListener('click',()=>{thresholds=Object.fromEntries(data.models.map(m=>[m.id,m.threshold]));render();});
$('export').addEventListener('click',()=>{const rows=[['model','split','threshold','precision','recall','f1','roc_auc','average_precision','tn','fp','fn','tp','illustrative_cost']];for(const m of data.models){const r=current(m);rows.push([m.name,$('split').value,thresholds[m.id],r.precision,r.recall,r.f1,r.auc,r.ap,r.tn,r.fp,r.fn,r.tp,r.cost]);}const url=URL.createObjectURL(new Blob([rows.map(r=>r.join(',')).join('\n')],{type:'text/csv'}));const a=document.createElement('a');a.href=url;a.download='evaluation_metrics.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
render();
}
