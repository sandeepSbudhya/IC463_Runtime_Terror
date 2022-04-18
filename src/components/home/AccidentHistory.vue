<template>
  <v-row>
    <v-col align="center">
      <v-card max-width="1000">
        <v-card-title>Accident history:</v-card-title>
        <v-data-table
          :headers="headers"
          :items="accidents"
          :items-per-page="10"
          class="elevation-1 py-auto"
          loading-text="Loading... Please wait"
        >
          <template v-slot:item.img_src="{ item }" class="py-auto">
            <v-img :src=" item.img_src" class="py-auto" style="width: 200px; height: 100px;" />
          </template>
        </v-data-table>
      </v-card>
    </v-col>
  </v-row>
</template>

<script>
// @ is an alias to /src

export default {
  name: "AccidentHistory",
  components: {},
  data() {
    return {
      headers: [
        {
          text: "Accident Image",
          align: "start",
          sortable: false,
          value: "img_src"
        },
        { text: "Location", value: "location.latitude", sortable: false },
        { text: "Date", value: "date", sortable: true }
      ]
    };
  },
  computed: {
    accidents() {
      return this.$root.accidents.filter(acc => {
        return !acc.is_reported;
      });
    }
  }
};
</script>
