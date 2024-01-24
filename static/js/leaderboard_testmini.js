function generateTable() {
  var data = score_table; // The variable from model_scores.js

  var table = '<table class="js-sort-table" id="results">';
  table += `<tr>
          <td class="js-sort-number"><strong>#</strong></td>
          <td class="js-sort"><strong>Model</strong></td>
          <td class="js-sort"><strong>Method</strong></td>
          <td class="js-sort"><strong>Source</strong></td>
          <td class="js-sort"><strong>Date</strong></td>
          <td class="js-sort-number"><strong><u>ALL</u></strong></td>
          <td class="js-sort-number"><strong>FQA</strong></td>
          <td class="js-sort-number"><strong>GPS</strong></td>
          <td class="js-sort-number"><strong>MWP</strong></td>
          <td class="js-sort-number"><strong>TQA</strong></td>
          <td class="js-sort-number"><strong>VQA</strong></td>
          <td class="js-sort-number"><strong>ALG</strong></td>
          <td class="js-sort-number"><strong>ARI</strong></td>
          <td class="js-sort-number"><strong>GEO</strong></td>
          <td class="js-sort-number"><strong>LOG</strong></td>
          <td class="js-sort-number"><strong>NUM</strong></td>
          <td class="js-sort-number"><strong>SCI</strong></td>
          <td class="js-sort-number"><strong>STA</strong></td>
      </tr>`;

      // sort data to make sure the best model is on top
      first_row = '-' // "Human Performance*"
      console.log(data);

      // get all keys in data
      var keys = Object.keys(data);

      // remove "Human Performance*" from keys
      var index = keys.indexOf(first_row);
      if (index > -1) {
        keys.splice(index, 1);
      }

      // add "Human Performance*" to the top of keys
      keys.unshift(first_row);

      console.log(keys);

      // for (var key in data) {
      for (var i=0; i<keys.length; i++) {
        var key = keys[i];
        console.log(key);
        var entry = data[key];

        table += '<tr>';
        table += `<td>${key}</td>`

        // for key = "1", "2", "3"
        top_ranks = ["1", "2", "3"]
        if (top_ranks.includes(key)) {
          table += `<td><b class="best-score-text">${entry.Model}</b></td>`;
          table += `<td>${entry.Method}</td>`;
          table += `<td><a href="${entry.Source}" class="ext-link" style="font-size: 16px;">Link</a></td>`;
          table += `<td>${entry.Date}</td>`;
          table += `<td><b class="best-score-text">${entry.ALL.toFixed(1).toString()}</b></td>`; // .toFixed(1): round to 1 decimal place
        }
        else {
          table += `<td><b>${entry.Model}</b></td>`;
          table += `<td>${entry.Method}</td>`;
          table += `<td><a href="${entry.Source}" class="ext-link" style="font-size: 16px;">Link</a></td>`;
          table += `<td>${entry.Date}</td>`;
          table += `<td><b>${entry.ALL.toFixed(1).toString()}</b></td>`; // .toFixed(1): round to 1 decimal place
        }          


        table += `<td>${entry.FQA.toFixed(1).toString()}</td>`;
        table += `<td>${entry.GPS.toFixed(1).toString()}</td>`;
        table += `<td>${entry.MWP.toFixed(1).toString()}</td>`;
        table += `<td>${entry.TQA.toFixed(1).toString()}</td>`;
        table += `<td>${entry.VQA.toFixed(1).toString()}</td>`;
        table += `<td>${entry.ALG.toFixed(1).toString()}</td>`;
        table += `<td>${entry.ARI.toFixed(1).toString()}</td>`;
        table += `<td>${entry.GEO.toFixed(1).toString()}</td>`;
        table += `<td>${entry.LOG.toFixed(1).toString()}</td>`;
        table += `<td>${entry.NUM.toFixed(1).toString()}</td>`;
        table += `<td>${entry.SCI.toFixed(1).toString()}</td>`;
        table += `<td>${entry.STA.toFixed(1).toString()}</td>`;
        table += '</tr>';
    }

  table += '</table>';
  document.getElementById('testmini_leaderboard').innerHTML = table; // Assuming you have a div with id 'container' where the table will be placed
}

// Call the function when the window loads
window.onload = generateTable;
